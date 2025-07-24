import pickle
import numpy as np
from tqdm import tqdm 

# with open('query_documents.pkl', 'rb') as f:
#     query_documents = pickle.load(f)

# all_entities = []
# for key in list(query_documents.keys()):
#     entities = query_documents[key]['meta_data']['entities']
#     for entity in entities:
#         all_entities.append(entity)


# all_entities = set(all_entities)

# print(len(all_entities))
# print(all_entities)



from search_document import load_documents, search_by_id, save_documents
from create_probing_questions import generate_probing_questions

#documents_list = ['KB0027217', 'KB0026014', 'KB0028512', 'KB0025749', 'KB0026668', 'KB0024122', 'KB0014146']

documents_list = ['KB0014146']

pickle_file = 'all_documents.pickle'
documents = load_documents(pickle_file)

new_content = '''

## Initial Probing Questions

Before proceeding with troubleshooting, please gather the following information:

1. **Are you using the Outlook desktop app, web version, or mobile app?**
2. **Have you confirmed that the Zoom Add-In is installed and visible in your Outlook ribbon?**
3. **Are you signed in to your Zoom account within the Outlook Add-In?**
4. **Is this your first time scheduling a Zoom meeting through Outlook?**
5. **According to the reply from user – Bot should follow the steps from KB**

---

## 1.0 Summary
Zoom is a video conferencing system and has an Outlook plugin which makes it easy to schedule a meeting. For further instructions on installing the Outlook plugin see: [Procedure - Zoom - Manual Install](https://sparc.service-now.com/kb_view.do?sysparm_article=KB0014104)Using the Zoom plugin a user can either Schedule a Meeting and/or start an Instant Zoom Meeting. This article outlines the process of scheduling a meeting and can be used by Enterprise Service Desk and Zoom users.
## 2.0 Environment
• Zoom video conferencing
## 3.0 How To Steps
NOTE: Some of the Zoom settings are locked down to create a consistent experience. To schedule a meeting using the Zoom Outlook plugin follow these steps:
1. Open Outlook > select the Calendar icon.
2. Navigate your calendar, right mouse click a time slot to create a New Meetingor select New Meeting from the Outlook ribbon.
3. Enter the parameters of the meeting and add participants
4. Select the Zoom Add-in from the ribbon to Add a Zoom Meeting
5. Select the three ellipsis (...) and select Zoom >Settings to modify the parameters of the Zoom Meeting. Once any modifications are made, select Update.
6. Send the meeting invite.
## 4.0 Escalation Procedure
For information on the escalation procedure, see the escalation section of [ESD Procedure - Zoom Support](https://sparc.service-now.com/kb_view.do?sysparm_article=KB0014055)
## 5.0 Links, References and Comments
• [KB0014110](https://sparc.service-now.com/kb_view.do?sysparm_article=KB0014110) - Master Article - Zoom - Meetings, Conferencing and Video Communications
• [ESD Procedure - Zoom Support](https://sparc.service-now.com/kb_view.do?sysparm_article=KB0014055)
• [Zoom Web Portal](https://gilead.zoom.us/)
• [Zoom Video Tutorials](https://support.zoom.us/hc/en-us/articles/206618765-Zoom-Video-Tutorials)
• [Zoom Scheduling Meetings](https://support.zoom.us/hc/en-us/articles/201362413-Scheduling-meetings)
[Back to top](#top)

'''


def view_documents_list(documents, documents_list):
    # Process only the specified number of documents
    for doc_id in tqdm(documents_list):
        # search by document id
        doc = search_by_id(documents, doc_id)

        try:

            #print(f"Doc Title {doc.get('title', '')}")

            #get content
            content = doc.get('content', '')
            title = doc.get('title', ''),
            #probing_questions = generate_probing_questions(content)

            print(f"Title {title}")
            #print(f"Probing Questions {probing_questions}")
            print("============================")
            print(f"Content {content}")

        except Exception as e:
            print(f"Doc ID does not exist: {doc_id}")


def process_documents_list(documents, documents_list):

    # Process only the specified number of documents
    for doc_id in tqdm(documents_list):
        # search by document id
        doc = search_by_id(documents, doc_id)

        try:

            #print(f"Doc Title {doc.get('title', '')}")

            #get content
            content = doc.get('content', '')
            title = doc.get('title', ''),
            #probing_questions = generate_probing_questions(content)

            print(f"Title {title}")
            #print(f"Probing Questions {probing_questions}")
            doc['content'] = new_content
            print("============================")
            # save the modified document 
            save_documents(documents, pickle_file, backup=False)

        except Exception as e:
            print(f"Doc ID does not exist: {doc_id}")


process_documents_list(documents, documents_list)

# see the processed content 
new_documents = load_documents(pickle_file)
view_documents_list(new_documents, documents_list)

