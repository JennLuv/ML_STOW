import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import joblib
import sys
import qrcode
import io
from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.realpath('/Users/Darren/OneDrive/Documents/Downloads/fp_python/fir-ml-9258b-firebase-adminsdk-391on-fe5321504b.json')

storage_bucket ='fir-ml-9258b.appspot.com'
firebase_admin.initialize_app(options={'storageBucket': storage_bucket})

# Reference to Firestore database
db = firestore.client()

# Retrieve data from Firestore
collection_ref = db.collection('sample')  # Replace 'your_collection' with your actual collection name

# Query documents in the collection
documents = collection_ref.get()

# Iterate over the documents and print their data
for document in documents:
    document_data = document.to_dict()
    # print(document_data)

df = pd.DataFrame(document_data)

data_sample = pd.DataFrame(df['message'].values.tolist())
# Extract the list of dictionaries from the 'message' key
message_list = document_data['message']


matrix = data_sample[['destinations', 'cargo_type', 'cargo_weight']].to_numpy()

rfc_model = joblib.load("rfc_model.joblib")
testrun3 = rfc_model.predict(matrix)

# Create a new DataFrame with the predicted locations and IDs
results_pd = pd.DataFrame({
    'id': data_sample['id'],
    'destinations': data_sample['destinations'],
    'cargo_type': data_sample['cargo_type'],
    'cargo_weight': data_sample['cargo_weight'],
    'Location': testrun3
})

# Save the results to a CSV file
results_pd.to_csv('results.csv', index=False)

matrix_data = pd.read_csv('results.csv')
data = matrix_data.values
data
sum_val = 50000
data =  np.save('results_modified.npy', data)


def execute_prediction(data, sum_val):
      # Assuming the data is already loaded into a numpy array called matrix
  data = np.load('results_modified.npy')

  selected_containers = None
  selected_ids = []
  while selected_containers is None:
      loc_1 = data[data[:, 4] == 1, :]
      loc_2 = data[data[:, 4] == 2, :]
      loc_3 = data[data[:, 4] == 3, :]
      loc_4 = data[data[:, 4] == 4, :]
      loc_1 = loc_1[np.random.choice(loc_1.shape[0], 1), :]
      loc_2 = loc_2[np.random.choice(loc_2.shape[0], 1), :]
      loc_3 = loc_3[np.random.choice(loc_3.shape[0], 1), :]
      loc_4 = loc_4[np.random.choice(loc_4.shape[0], 1), :]


      total_weight = loc_1[:, 3] + loc_2[:, 3] + loc_3[:, 3] + loc_4[:, 3]
      if total_weight == sum_val + 1000 or sum_val - 1000:
          selected_containers = np.concatenate((loc_1, loc_2, loc_3, loc_4), axis=0)

  # Remove the selected rows from the data
  selected_ids.extend(selected_containers[:, 0].tolist())
  data = data[~np.isin(data[:, 0], selected_ids), :]

  # Put the selected containers into a matrix
  tier = selected_containers[:, [0, 1, 2, 3, 4]]

  # Save the modified data to a numpy array
  np.save('results_modified.npy', data)

  return tier

#how many tiers do you want?
y = 1
loop = y*2
results = []
for i in range(loop):
    results.append(execute_prediction(data, sum_val))

for i, matrix in enumerate(results):
    if i % 2 == 1:
        matrix[0,4] = 5
        matrix[1,4] = 6
        matrix[2,4] = 7
        matrix[3,4] = 8

# Initialize Firebase Storage client
storage_client = storage.Client()

# Store the results in Firestore and Firebase Storage
db = firestore.client()
bucket = storage_client.get_bucket('fir-ml-9258b.appspot.com')

for i, result in enumerate(results):
    for j, container in enumerate(result):
        container_id = int(container[0])
        doc_name = u'containers{}'.format(container_id + 1)
        doc_ref = db.collection(u'results').document(doc_name)
        qr_code_data = doc_name  # Set the QR code data as the document name

        # Generate the QR code image
        qr_code = qrcode.make(qr_code_data)

        # Create a BytesIO object to store the QR code image
        qr_code_buffer = io.BytesIO()
        qr_code.save(qr_code_buffer)
        qr_code_buffer.seek(0)

        # Upload the QR code image to Firebase Storage
        qr_code_filename = f'{doc_name}.png'
        blob = bucket.blob(qr_code_filename)
        blob.upload_from_file(qr_code_buffer, content_type='image/png')

        # Get the public URL of the uploaded QR code image
        qr_code_url = blob.public_url

        # Set the QR code URL in Firestore
        doc_ref.set({
            u'id': container_id,
            u'destinations': int(container[1]),
            u'cargo_type': str(container[2]),
            u'cargo_weight': int(container[3]),
            u'location': int(container[4]),
        })

        print("QR Code URL:", qr_code_url)


print(results)