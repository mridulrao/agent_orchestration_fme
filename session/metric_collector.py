import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
    MetricsCollectedEvent,
)

# Configure logging
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Comprehensive metrics tracker for STT, LLM, and TTS performance"""
    
    def __init__(self):
        self.stt_metrics: List[Dict] = []
        self.llm_metrics: List[Dict] = []
        self.tts_metrics: List[Dict] = []
        self.eou_metrics: List[Dict] = []
        self.usage_collector = metrics.UsageCollector()
        
        # Performance tracking
        self.conversation_start_time = time.time()
        self.total_conversations = 0
        
        # Dynamically check available metrics classes
        self._available_metrics = self._check_available_metrics()
        
    def _check_available_metrics(self) -> Dict[str, List]:
        """Check which metrics classes are available in the current LiveKit version"""
        available = {
            'stt_classes': [],
            'llm_classes': [],
            'tts_classes': [],
            'eou_classes': []
        }
        
        # Check for STT metrics classes
        for class_name in ['STTMetrics', 'PipelineSTTMetrics']:
            if hasattr(metrics, class_name):
                available['stt_classes'].append(getattr(metrics, class_name))
        
        # Check for LLM metrics classes  
        for class_name in ['LLMMetrics', 'PipelineLLMMetrics']:
            if hasattr(metrics, class_name):
                available['llm_classes'].append(getattr(metrics, class_name))
        
        # Check for TTS metrics classes
        for class_name in ['TTSMetrics', 'PipelineTTSMetrics']:
            if hasattr(metrics, class_name):
                available['tts_classes'].append(getattr(metrics, class_name))
        
        # Check for EOU metrics classes
        for class_name in ['EOUMetrics', 'PipelineEOUMetrics']:
            if hasattr(metrics, class_name):
                available['eou_classes'].append(getattr(metrics, class_name))
        
        logger.info(f"Available metrics classes: {[cls.__name__ for classes in available.values() for cls in classes]}")
        return available
        
    def collect_metrics(self, metric_event: metrics.AgentMetrics):
        """Collect and categorize different types of metrics"""
        
        # Use the built-in usage collector
        self.usage_collector.collect(metric_event)
        
        # Convert metrics to dictionary for easy storage/analysis
        try:
            # Try asdict first (for dataclass instances)
            metric_dict = asdict(metric_event)
        except (TypeError, AttributeError):
            # Fall back to manual conversion for non-dataclass instances
            metric_dict = self._convert_to_dict(metric_event)
        
        metric_dict['collected_at'] = time.time()
        
        # Get the metric type name for robust classification
        metric_type = type(metric_event).__name__
        
        # Categorize metrics using both isinstance check and string matching
        is_stt = (any(isinstance(metric_event, cls) for cls in self._available_metrics['stt_classes']) or 
                 'STT' in metric_type)
        
        is_llm = (any(isinstance(metric_event, cls) for cls in self._available_metrics['llm_classes']) or 
                 'LLM' in metric_type)
        
        is_tts = (any(isinstance(metric_event, cls) for cls in self._available_metrics['tts_classes']) or 
                 'TTS' in metric_type)
        
        is_eou = (any(isinstance(metric_event, cls) for cls in self._available_metrics['eou_classes']) or 
                 'EOU' in metric_type)
        
        if is_stt:
            self.stt_metrics.append(metric_dict)
            self._log_stt_metrics(metric_event)
            
        elif is_llm:
            self.llm_metrics.append(metric_dict)
            self._log_llm_metrics(metric_event)
            
        elif is_tts:
            self.tts_metrics.append(metric_dict)
            self._log_tts_metrics(metric_event)
            
        elif is_eou:
            self.eou_metrics.append(metric_dict)
            self._log_eou_metrics(metric_event)
        
        # Log the metric type for debugging
        logger.debug(f"Collected metric of type: {metric_type}")
    
    def _convert_to_dict(self, obj) -> Dict:
        """Convert metrics object to dictionary manually"""
        if hasattr(obj, '__dict__'):
            # Use object's __dict__ if available
            return dict(obj.__dict__)
        else:
            # Extract common attributes based on metric type
            result = {}
            
            # Common attributes for all metrics
            common_attrs = ['request_id', 'timestamp', 'label', 'error']
            for attr in common_attrs:
                if hasattr(obj, attr):
                    result[attr] = getattr(obj, attr)
            
            # STT specific attributes
            if 'STT' in str(type(obj)):
                stt_attrs = ['duration', 'audio_duration', 'streamed', 'sequence_id']
                for attr in stt_attrs:
                    if hasattr(obj, attr):
                        result[attr] = getattr(obj, attr)
            
            # LLM specific attributes
            elif 'LLM' in str(type(obj)):
                llm_attrs = ['ttft', 'duration', 'cancelled', 'completion_tokens', 
                           'prompt_tokens', 'total_tokens', 'tokens_per_second', 'sequence_id']
                for attr in llm_attrs:
                    if hasattr(obj, attr):
                        result[attr] = getattr(obj, attr)
            
            # TTS specific attributes
            elif 'TTS' in str(type(obj)):
                tts_attrs = ['ttfb', 'duration', 'audio_duration', 'cancelled', 
                           'characters_count', 'streamed', 'sequence_id']
                for attr in tts_attrs:
                    if hasattr(obj, attr):
                        result[attr] = getattr(obj, attr)
            
            # EOU specific attributes
            elif 'EOU' in str(type(obj)):
                eou_attrs = ['end_of_utterance_delay', 'transcription_delay', 'sequence_id']
                for attr in eou_attrs:
                    if hasattr(obj, attr):
                        result[attr] = getattr(obj, attr)
            
            return result
    
    def _log_stt_metrics(self, stt_metric):
        """Log STT-specific metrics"""
        try:
            duration = getattr(stt_metric, 'duration', 0)
            audio_duration = getattr(stt_metric, 'audio_duration', 0)
            streamed = getattr(stt_metric, 'streamed', False)
            
            logger.info(
                f"STT - Duration: {duration:.2f}s, "
                f"Audio: {audio_duration:.2f}s, "
                f"Streamed: {streamed}"
            )
        except Exception as e:
            logger.warning(f"Error logging STT metrics: {e}")
    
    def _log_llm_metrics(self, llm_metric):
        """Log LLM-specific metrics"""
        try:
            ttft = getattr(llm_metric, 'ttft', 0)
            duration = getattr(llm_metric, 'duration', 0)
            tokens_per_second = getattr(llm_metric, 'tokens_per_second', 0)
            prompt_tokens = getattr(llm_metric, 'prompt_tokens', 0)
            completion_tokens = getattr(llm_metric, 'completion_tokens', 0)
            
            logger.info(
                f"LLM - TTFT: {ttft:.2f}s, "
                f"Duration: {duration:.2f}s, "
                f"Tokens/sec: {tokens_per_second:.2f}, "
                f"Tokens: {prompt_tokens}/{completion_tokens}"
            )
        except Exception as e:
            logger.warning(f"Error logging LLM metrics: {e}")
    
    def _log_tts_metrics(self, tts_metric):
        """Log TTS-specific metrics"""
        try:
            ttfb = getattr(tts_metric, 'ttfb', 0)
            duration = getattr(tts_metric, 'duration', 0)
            characters_count = getattr(tts_metric, 'characters_count', 0)
            
            logger.info(
                f"TTS - TTFB: {ttfb:.2f}s, "
                f"Duration: {duration:.2f}s, "
                f"Characters: {characters_count}"
            )
        except Exception as e:
            logger.warning(f"Error logging TTS metrics: {e}")
    
    def _log_eou_metrics(self, eou_metric):
        """Log End-of-Utterance metrics"""
        try:
            end_of_utterance_delay = getattr(eou_metric, 'end_of_utterance_delay', 0)
            transcription_delay = getattr(eou_metric, 'transcription_delay', 0)
            
            logger.info(
                f"EOU - Delay: {end_of_utterance_delay:.2f}s, "
                f"Transcription Delay: {transcription_delay:.2f}s"
            )
        except Exception as e:
            logger.warning(f"Error logging EOU metrics: {e}")
    
    def calculate_conversation_latency(self) -> Optional[float]:
        """Calculate total conversation latency"""
        if not (self.eou_metrics and self.llm_metrics and self.tts_metrics):
            return None
        
        latest_eou = self.eou_metrics[-1]
        latest_llm = self.llm_metrics[-1]
        latest_tts = self.tts_metrics[-1]
        
        total_latency = (
            latest_eou.get('end_of_utterance_delay', 0) +
            latest_llm.get('ttft', 0) +
            latest_tts.get('ttfb', 0)
        )
        
        return total_latency
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        usage_summary = self.usage_collector.get_summary()
        session_duration = time.time() - self.conversation_start_time
        total_latency = self.calculate_conversation_latency()
        
        return {
            'session_metrics': {
                'session_duration': session_duration,
                'total_conversation_latency': total_latency,
            },
            'usage_summary': {
                'llm_prompt_tokens': usage_summary.llm_prompt_tokens,
                'llm_completion_tokens': usage_summary.llm_completion_tokens,
                'total_llm_tokens': usage_summary.llm_prompt_tokens + usage_summary.llm_completion_tokens,
                'tts_characters_count': usage_summary.tts_characters_count,
                'stt_audio_duration': usage_summary.stt_audio_duration,
            },
            'component_counts': {
                'stt_requests': len(self.stt_metrics),
                'llm_requests': len(self.llm_metrics),
                'tts_requests': len(self.tts_metrics),
                'eou_events': len(self.eou_metrics),
            },
            'average_performance': self._calculate_averages(),
        }
    
    def _calculate_averages(self) -> Dict:
        """Calculate average performance metrics"""
        avg_metrics = {}
        
        if self.stt_metrics:
            avg_metrics['stt'] = {
                'avg_duration': sum(m.get('duration', 0) for m in self.stt_metrics) / len(self.stt_metrics),
                'avg_audio_duration': sum(m.get('audio_duration', 0) for m in self.stt_metrics) / len(self.stt_metrics),
            }
        
        if self.llm_metrics:
            avg_metrics['llm'] = {
                'avg_ttft': sum(m.get('ttft', 0) for m in self.llm_metrics) / len(self.llm_metrics),
                'avg_duration': sum(m.get('duration', 0) for m in self.llm_metrics) / len(self.llm_metrics),
                'avg_tokens_per_second': sum(m.get('tokens_per_second', 0) for m in self.llm_metrics) / len(self.llm_metrics),
            }
        
        if self.tts_metrics:
            avg_metrics['tts'] = {
                'avg_ttfb': sum(m.get('ttfb', 0) for m in self.tts_metrics) / len(self.tts_metrics),
                'avg_duration': sum(m.get('duration', 0) for m in self.tts_metrics) / len(self.tts_metrics),
            }
        
        return avg_metrics
    
    def export_metrics_to_json(self, filename: str):
        """Export all collected metrics to JSON file"""
        export_data = {
            'summary': self.get_performance_summary(),
            'detailed_metrics': {
                'stt_metrics': self.stt_metrics,
                'llm_metrics': self.llm_metrics,
                'tts_metrics': self.tts_metrics,
                'eou_metrics': self.eou_metrics,
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Detailed metrics exported to {filename}")