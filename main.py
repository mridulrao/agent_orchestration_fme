import logging
import os
import json
import asyncio
import time
import datetime
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.voice import AgentSession
from livekit.plugins import openai, silero
from livekit.agents import BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import noise_cancellation
from livekit import rtc, api
from livekit.agents import metrics, MetricsCollectedEvent

# agents
from agents.service_agent import ServiceAgent
from agents.truu_agent import TruUAgent
from agents.gadi_agent import GADIAgent
from agents.laptop_troubleshooting_agent import LaptopTroubleshootinAgent

# session
from session.user_data import UserData
from session.vva_session_info import VVASessionInfo
from session.monitor_inactivity import InactivityMonitor
from session.post_call_processing import PostCallProcessor

# tooling tracker
from tooling.function_tool_analytics import FunctionToolTracker
from tooling.helper_functions import validate_dtmf_employee_id
from tooling.helper_functions import verify_employee_dtmf

# db
from db.handler import DatabaseHandler
from db.operations import DatabaseOperations

# snow
from servicenow.servicenow_auth import ServiceNow, ServiceNowAuth
auth = ServiceNowAuth(
            username="svc_worknext_dev",
            password="-f->sn)pi^5w!-B_7E@1JlzD&vQnF3(s.D82b*A",
            instance_name="fmcnadev"
        )

# rag
from rag.unified_rag_agent_local_follow_up import KnowledgeQueryEngine

# metric
# from session.metric_collector import MetricsTracker

load_dotenv()

logger = logging.getLogger("IT-support-triage")
logger.setLevel(logging.INFO)

# Configuration
lancedb_uri = os.environ.get("KB_LANCEDB_URI", "./vector_index")
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
snow_client_id = os.environ.get("SNOW_CLIENT_ID")
snow_client_secret = os.environ.get("SNOW_CLIENT_SECRET")
snow_env = os.environ.get("SNOW_ENV")
snow_base_url = os.environ.get("SNOW_BASE_URL")

# ServiceNow config (adjust as needed)
TEST_CONFIG = {
    "client_id": snow_client_id,
    "client_secret": snow_client_secret,
    "env": snow_env,
    "base_url": snow_base_url,
}

azure_storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT")
azure_storage_key = os.environ.get("AZURE_STORAGE_KEY")
azure_container_name = os.environ.get("AZURE_CONTAINER_NAME")

vva_phone_no = os.environ.get("VVA_PHONE_NUMBER")


async def prewarm(proc: JobProcess):
    """Prewarm function to initialize shared resources"""
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.07, min_silence_duration=0.7, activation_threshold=0.5
    )
    proc.userdata["query_engine"] = KnowledgeQueryEngine(
        openai_api_key=openai_api_key,
        use_openai_embeddings=True,
        openai_embedding_model="text-embedding-3-small",
    )

    config = Config(
        client_id=TEST_CONFIG["client_id"],
        client_secret=TEST_CONFIG["client_secret"],
        env=TEST_CONFIG["env"],
        base_url=TEST_CONFIG["base_url"],
        ssl_verify=False,
    )
    proc.userdata["snow"] = ServiceNow(auth)

    # Initialize database handler and operations with retry logic and connection pooling
    logger.info("Initializing database handler and operations in prewarm...")
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Database connection attempt {attempt + 1}/{max_retries}")
            db_handler = DatabaseHandler()

            # Connect with explicit timeout and pooling configuration
            await db_handler.connect()

            # Test the connection with a simple query
            # async with db_handler.get_connection() as client:
            #     # Verify connection is working
            #     await client.connect()
            #     logger.info("Database connection verified successfully")

            proc.userdata["db_handler"] = db_handler
            proc.userdata["db_operations"] = DatabaseOperations(db_handler)
            logger.info(
                f"Database initialized successfully in prewarm on attempt {attempt + 1}"
            )
            break

        except Exception as e:
            logger.warning(
                f"Database connection attempt {attempt + 1} failed: {str(e)}"
            )

            if attempt < max_retries - 1:
                logger.info(f"Retrying database connection in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(
                    f"Failed to initialize database after {max_retries} attempts"
                )
                # Set to None to trigger fallback in entrypoint
                proc.userdata["db_handler"] = None
                proc.userdata["db_operations"] = None


def prewarm_sync(proc: JobProcess):
    """Synchronous wrapper for async prewarm function"""

    async def _async_prewarm():
        await prewarm(proc)

    # Run the async prewarm function
    asyncio.run(_async_prewarm())


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the IT support triage system"""
    await ctx.connect()

    ctx.log_context_fields = {
        "room_name": ctx.room.name,
        "user_id": "it_support_user",
    }

    # Initialize user data with VVA session info
    userdata = UserData(ctx=ctx)

    # Process org_id from VVA dispatch rule if available
    if hasattr(ctx.job, "metadata") and ctx.job.metadata:
        org_id = userdata.get_org_id(ctx.job.metadata)
        userdata.org_id = org_id
        userdata.transfer_number = userdata.get_transfer_number(ctx.job.metadata)

    # Process call details from room name
    room_call_id, phone_number = userdata.get_call_id_and_phone_number(ctx.room.name)
    userdata.phone_number = phone_number
    userdata.room_name = ctx.room.name
    # Note: We'll create a proper UUID call_id in the database, not use the room name call_id

    # Wait for participant and store details
    participant = await ctx.wait_for_participant()
    userdata.participant_details = participant.identity

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_file = f"{ctx.room.name}_{timestamp}_whisper"

    # try:
    #     request = api.RoomCompositeEgressRequest(
    #         room_name=ctx.room.name,
    #         layout="speaker",
    #         audio_only=True,
    #         file_outputs=[
    #             api.EncodedFileOutput(
    #                 file_type=api.EncodedFileType.OGG,
    #                 filepath=f"{recording_file}.ogg",
    #                 azure=api.AzureBlobUpload(
    #                     account_name=azure_storage_account,
    #                     account_key=azure_storage_key,
    #                     container_name=azure_container_name,
    #                 ),
    #             )
    #         ],
    #     )

    #     logger.debug(f"Recording request payload: {request}")

    #     recording_info = await ctx.api.egress.start_room_composite_egress(request)

    #     logger.info(f"Egress started: {json.dumps(recording_info, default=str)}")

    # except Exception as e:
    #     logger.error("Failed to start egress recording", exc_info=True)

    # Initialize ServiceNow instance
    try:
        userdata.snow = ctx.proc.userdata.get("snow")
        print(f"Passes SNOW ==> {userdata.snow}")

        if userdata.snow:
            # Initialize the session if ServiceNow client exists
            try:
                await userdata.snow._get_session()
                logger.info("ServiceNow client initialized successfully")
            except Exception as session_error:
                logger.warning(
                    f"Could not initialize ServiceNow session: {session_error}"
                )
        else:
            logger.warning("ServiceNow client not found in proc.userdata")
    except Exception as e:
        logger.warning(f"Could not initialize ServiceNow: {e}")

    # Initialize database handler and operations
    try:
        logger.info("Retrieving database operations from proc.userdata...")
        logger.info(
            f"Available keys in proc.userdata: {list(ctx.proc.userdata.keys())}"
        )

        userdata.db_handler = ctx.proc.userdata.get("db_handler")
        userdata.db_operations = ctx.proc.userdata.get("db_operations")

        # Debug logging to check database operations initialization
        logger.info(f"Database handler retrieved: {userdata.db_handler is not None}")
        logger.info(
            f"Database operations retrieved: {userdata.db_operations is not None}"
        )

        if userdata.db_handler is None or userdata.db_operations is None:
            logger.warning(
                "Database handler or operations is None - creating fallback..."
            )
            # Fallback: create database operations directly only if prewarm failed
            max_fallback_retries = 2
            fallback_delay = 1.0

            for fallback_attempt in range(max_fallback_retries):
                try:
                    logger.info(
                        f"Fallback database connection attempt {fallback_attempt + 1}/{max_fallback_retries}"
                    )
                    fallback_db_handler = DatabaseHandler()
                    await fallback_db_handler.connect(
                        max_retries=2
                    )  # Fewer retries for fallback

                    # Test connection before proceeding
                    # async with fallback_db_handler.get_connection() as client:
                    #     await client.connect()
                    #     logger.info("Fallback database connection verified")

                    userdata.db_handler = fallback_db_handler
                    userdata.db_operations = DatabaseOperations(fallback_db_handler)
                    logger.info("Fallback database operations created successfully")
                    break

                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback attempt {fallback_attempt + 1} failed: {str(fallback_error)}"
                    )
                    if fallback_attempt < max_fallback_retries - 1:
                        logger.info(f"Retrying fallback in {fallback_delay} seconds...")
                        await asyncio.sleep(fallback_delay)
                        fallback_delay *= 1.5
                    else:
                        logger.error(
                            f"Failed to create fallback database operations after {max_fallback_retries} attempts"
                        )

        # Test database connection without trying to reconnect
        if userdata.db_handler:
            try:
                # Just test the connection without trying to reconnect
                async with userdata.db_handler.get_connection() as client:
                    # Try a simple query to test connection
                    test_result = await client.voice_virtual_agent_calls.find_first()
                    logger.info("Database connection test successful")
            except Exception as test_error:
                logger.error(f"Database connection test failed: {test_error}")
        else:
            logger.warning("Database handler not available for testing")

        # Create call record in database with minimal data (will be updated later)
        logger.info("=== CALL CREATION SECTION START ===")
        logger.info(
            f"Attempting to create call record. db_operations={userdata.db_operations is not None}"
        )
        logger.info(f"Phone number: {userdata.phone_number}")
        logger.info(f"Recording file: {recording_file}")

        if userdata.db_operations:
            try:
                call_data = {
                    "caller": userdata.phone_number,  # Phone number (available at start)
                    "user_name": None,  # Will be updated after employee verification
                    "user_sys_id": None,  # Will be updated after employee verification
                    "incident_ticket_ids": [],  # Will be updated as tickets are created
                    "ticket_id": None,  # Will be updated when first ticket is created
                    "recording_url": f"{recording_file}.ogg",  # File name from recording
                    "status": "ongoing",  # Will be updated to "ended" when call completes
                    "called_at": str(vva_phone_no)
                }
                logger.info(f"Call data prepared: {call_data}")

                logger.info(f"About to call create_vva_call with data: {call_data}")
                created_call_id = await userdata.db_operations.create_vva_call(
                    call_data
                )
                logger.info(f"create_vva_call returned: {created_call_id}")
                if created_call_id:
                    logger.info(
                        f"Call record created in database with ID: {created_call_id}"
                    )
                    # Set the call_id to use the database-generated UUID for all future operations
                    userdata.call_id = created_call_id
                    logger.info(
                        f"Using database-generated UUID as call_id: {created_call_id}"
                    )
                else:
                    logger.warning("Failed to create call record in database")
            except Exception as call_creation_error:
                logger.warning(f"Could not create call record: {call_creation_error}")
                logger.error(f"Call creation error details: {str(call_creation_error)}")
        else:
            logger.warning(
                f"Database operations not available for call creation. db_operations={userdata.db_operations is not None}"
            )

        logger.info("=== CALL CREATION SECTION END ===")

    except Exception as e:
        logger.warning(f"Could not initialize database handler: {e}")

    # Initialize query engine
    try:
        userdata.query_engine = ctx.proc.userdata.get(
            "query_engine",
        )
    except Exception as e:
        logger.warning(f"Could not initialize KnowledgeQueryEngine: {e}")

    # initialize function tool tracker
    userdata.function_tool_tracker = FunctionToolTracker()

    # Initialize all agents
    service_agent = ServiceAgent()
    truu_agent = TruUAgent()
    gadi_agent = GADIAgent()
    laptop_troubleshooting_agent = LaptopTroubleshootinAgent()

    # Register all agents in the userdata
    userdata.personas.update(
        {
            "service": service_agent,
            "truu": truu_agent,
            "gadi": gadi_agent,
            "laptop": laptop_troubleshooting_agent,
        }
    )

    # Create session with enhanced userdata
    session = AgentSession[UserData](userdata=userdata, max_tool_steps=7)

    # Ensure database operations and call_id are properly set in session
    if userdata.db_operations and not session.userdata.db_operations:
        session.userdata.db_operations = userdata.db_operations
        logger.info("Database operations manually set in session")

    if userdata.db_handler and not session.userdata.db_handler:
        session.userdata.db_handler = userdata.db_handler
        logger.info("Database handler manually set in session")

    if userdata.call_id and not session.userdata.call_id:
        session.userdata.call_id = userdata.call_id
        logger.info(f"Call ID manually set in session: {userdata.call_id}")

    # Debug logging to check if database operations are available in session
    logger.info(
        f"Session userdata db_operations available: {session.userdata.db_operations is not None}"
    )
    logger.info(
        f"Session userdata db_handler available: {session.userdata.db_handler is not None}"
    )

    # Register conversation listeners early to ensure all transcripts are captured
    try:
        # Use session.userdata instead of userdata to ensure database operations are available
        logger.info(
            f"Registering conversation listeners. session.userdata.db_operations={session.userdata.db_operations is not None}"
        )
        session.userdata.register_conversation_listeners(session)
        logger.info(
            "Conversation listeners registered for real-time transcript capture"
        )
    except Exception as e:
        logger.warning(f"Could not register conversation listeners: {e}")
        logger.error(f"Conversation listener registration error details: {str(e)}")

    # Initialize inactivity monitor
    try:
        inactivity_monitor = InactivityMonitor(
            session=session,
            prompt_timeout=20.0,
            hangup_timeout=120.0,
            room_name=ctx.room.name,
        )
        await inactivity_monitor.start(ctx)
        userdata.inactivity_monitor = inactivity_monitor
    except Exception as e:
        logger.warning(f"Could not initialize inactivity monitor: {e}")

    #  background noise
    # background_audio = BackgroundAudioPlayer(
    #     # general audio looping in background
    #     # ambient_sound = AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume = 0.8),
    #     # audio for while thinking
    #     thinking_sound = [
    #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume = 0.2),
    #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume = 0.1),
    #     ],
    # )

    # await background_audio.start(room = ctx.room, agent_session = session)

    # Start session with the primary service agent
    await session.start(
        agent=service_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony()
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # Initial greeting
    await session.say("Thank you for calling Enterprise Service Desk. I am Paula, To assist you please provide your Employee ID verbally, or enter it using the dial pad, followed by pound or hash key")

    # Update call status to "ongoing" after VVA starts
    if session.userdata.db_operations and session.userdata.call_id:
        try:
            await session.userdata.db_operations.update_call_status(
                session.userdata.call_id, "ongoing"
            )
            logger.info(
                f"Call status updated to 'ongoing' for call_id: {session.userdata.call_id}"
            )
        except Exception as e:
            logger.warning(f"Could not update call status to 'ongoing': {e}")

    # Define cleanup functions
    async def on_participant_disconnected(participant):
        logger.info(f"Participant disconnected: {participant.identity}")

        ctx.shutdown(reason="customer-ended-call")

    # Create a synchronous wrapper function
    def on_participant_disconnected_sync(participant):
        import asyncio

        # Create an asyncio task from the async function
        asyncio.create_task(on_participant_disconnected(participant))

    # Define the shutdown callback with the reason parameter
    async def shutdown_task(reason: str):
        logger.debug(f"INITIATING EXIT CALL. Reason: {reason}")

        try:
            # Initialize the processor
            processor = PostCallProcessor(session.userdata)
            results = await processor.process_call()

            # Access specific summary fields
            if results.summary_data:
                print(f"Issue: {results.summary_data['issue_summary']}")
                print(f"Customer: {results.summary_data['customer_details']}")
                print(f"Follow-up needed: {results.summary_data['follow_up_needed']}")

            # Get clean summary details
            summary_details = processor.get_summary_details()
            print(f"Resolution status: {summary_details['resolution_status']}")

            # Get complete report
            report = processor.get_simple_report()
            full_summary = report["processing"]["summary"]["data"]
            print(f"Full summary {full_summary}")

            # Access just the outputs
            outputs = processor.get_detailed_outputs()
            call_summary = outputs["call_summary"]
            print(f"Call summary {call_summary}")

            # Save summary and sentiment to database
            if session.userdata.db_operations and session.userdata.call_id:
                try:
                    # Extract summary and sentiment from the results
                    call_summary = (
                        json.dumps(results.summary_data)
                        if results.summary_data
                        else json.dumps({})
                    )
                    sentiment = (
                        results.summary_data.get("sentiment", "neutral")
                        if results.summary_data
                        else "neutral"
                    )

                    # Update the call record with summary and sentiment
                    success = await session.userdata.db_operations.update_vva_call_summary_and_sentiment(
                        call_id=session.userdata.call_id,
                        summary=call_summary,
                        sentiment=sentiment,
                    )

                    if success:
                        logger.info(
                            f"Call summary and sentiment saved to database for call_id: {session.userdata.call_id}"
                        )
                    else:
                        logger.warning(
                            f"Failed to save call summary and sentiment to database for call_id: {session.userdata.call_id}"
                        )

                except Exception as db_error:
                    logger.error(f"Error saving call summary to database: {db_error}")
            else:
                logger.warning(
                    f"Database operations or call_id not available for saving summary. db_operations={session.userdata.db_operations is not None}, call_id={session.userdata.call_id}"
                )

        except Exception as e:
            logger.error(f"Failed to process post-call data: {str(e)}")
            return None

        # End the call in database
        if session.userdata.db_operations and session.userdata.call_id:
            try:
                # Update call status to "ended"
                await session.userdata.db_operations.update_call_status(
                    session.userdata.call_id, "ended"
                )
                logger.info(
                    f"Call status updated to 'ended' for call_id: {session.userdata.call_id}"
                )

                # Determine the recording URL from the session
                recording_url = None
                if (
                    hasattr(session.userdata, "room_name")
                    and session.userdata.room_name
                ):
                    # Construct recording URL from room name and timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    recording_url = (
                        f"{session.userdata.room_name}_{timestamp}_whisper.ogg"
                    )

                await session.userdata.db_operations.end_vva_call(
                    call_id=session.userdata.call_id,
                    ended_reason="session_ended",  # "agent_session_ended" or "customer-ended-call"
                    ended_at=datetime.datetime.now(),
                    recording_url=recording_url,
                )
                logger.info(
                    f"Call ended in database for call_id: {session.userdata.call_id} with reason: {reason}"
                )
            except Exception as db_error:
                logger.error(f"Error ending call in database: {db_error}")
        else:
            logger.warning(
                f"Database operations or call_id not available for ending call. db_operations={session.userdata.db_operations is not None}, call_id={session.userdata.call_id}"
            )

        # close the snow and db session
        if session.userdata.snow:
            await session.userdata.snow.close()

    @ctx.room.on("sip_dtmf_received")
    def dtmf_received(dtmf: rtc.SipDTMF):
        if userdata.dtmf_input is None:
            userdata.dtmf_input = ""

        if dtmf.digit == "#":  # Process when # is pressed
            if userdata.dtmf_input:
                logging.info(f"Processing DTMF input: {userdata.dtmf_input}")

                asyncio.create_task(process_dtmf_input(userdata.dtmf_input))
            else:
                logging.info("No DTMF input to process")

        elif len(userdata.dtmf_input) < 6:
            userdata.dtmf_input += str(dtmf.digit)
            logging.info(
                f"DTMF input updated: {userdata.dtmf_input} (length: {len(userdata.dtmf_input)})"
            )
        else:
            logging.info(
                f"DTMF input ignored - already at maximum length of 6 digits: {userdata.dtmf_input}"
            )

    async def process_dtmf_input(employeeId):
        try:
            await session.say("Please wait while I verify your employee ID")
            result = await verify_employee_dtmf(userdata, employeeId)
            print(result)

            # Check if verification was successful
            if result.get("error"):
                # Handle different types of errors
                error_msg = result["error"]
                if "not found" in error_msg.lower():
                    await session.say("I couldn't find an employee with that ID. Please check your employee number and try again.")
                else:
                    await session.say("There was an issue verifying your employee ID. Please try again or contact support.")
                logging.warning(
                    f"Employee verification failed for {employeeId}: {error_msg}"
                )
                return

            # Success case
            full_name = result.get("full_name")
            user_sys_id = result.get("sys_id")
            userdata.user_sys_id = user_sys_id
            userdata.full_name = full_name

            # Update call record with user information
            if (
                session.userdata.db_operations
                and session.userdata.call_id
                and full_name
                and user_sys_id
            ):
                try:
                    await session.userdata.db_operations.update_call_user_info(
                        call_id=session.userdata.call_id,
                        user_name=full_name,
                        user_sys_id=user_sys_id,
                    )
                    logger.info(f"Call record updated with user info: {full_name}")
                except Exception as e:
                    logger.warning(f"Could not update call record with user info: {e}")

            if full_name:
                await session.say(f"Hi {full_name}, how can I assist you today?")
            else:
                await session.say("Employee verification was incomplete. Please try again.")

        except Exception as e:
            logging.error(f"Error processing DTMF input: {e}")
            await session.say("I'm sorry, there was a technical issue. Please try again.")

    # metrics_tracker = MetricsTracker()

    # @session.on("metrics_collected")
    # def on_metrics_collected(ev: MetricsCollectedEvent):
    #     metrics_tracker.collect_metrics(ev.metrics)
    #     metrics.log_metrics(ev.metrics)

    # async def log_comprehensive_usage():
    #     usage_summary = metrics_tracker.usage_collector.get_summary()
    #     logger.info(f"Usage: {usage_summary}")

    #     # Enhanced performance summary
    #     performance_summary = metrics_tracker.get_performance_summary()
    #     logger.info(f"Performance Summary: {json.dumps(performance_summary, indent=2, default=str)}")

    #     # Export detailed metrics
    #     timestamp = int(time.time())
    #     metrics_tracker.export_metrics_to_json(f"session_metrics_{timestamp}.json")

    # # Replace your existing shutdown callback
    # ctx.add_shutdown_callback(log_comprehensive_usage)

    ctx.room.on("participant_disconnected", on_participant_disconnected_sync)
    ctx.add_shutdown_callback(shutdown_task)


def main():
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_sync,
            job_memory_warn_mb=2000,
            shutdown_process_timeout=80.0,
            initialize_process_timeout=540,
            num_idle_processes=1,
        )
    )


if __name__ == "__main__":
    main()


"""

New updates - 
Ticket number is fast -- done 


DB Tables 
Data konsa store krna hai 



"""
