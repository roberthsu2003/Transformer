import json
from datasets import Dataset, DatasetDict

def convert_to_cmrc2018(input_file, output_file):
    """Converts a DRCD JSON file to cmrc2018 format."""

    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        drcd_data = json.load(f)['data']
        for article in drcd_data:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    cmrc_example = {
                        'id': qa['id'],
                        'context': paragraph['context'],
                        'question': qa['question'],
                        'answers': {
                            'text': [qa['answers'][0]['text']],  # Only take the first answer
                            'answer_start': [qa['answers'][0]['answer_start']]
                        }
                    }
                    data.append(cmrc_example)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'data': data}, f, ensure_ascii=False, indent=4)

# Convert your files
convert_to_cmrc2018('DRCD_training.json', 'cmrc2018_train.json')
convert_to_cmrc2018('DRCD_dev.json', 'cmrc2018_dev.json')
convert_to_cmrc2018('DRCD_test.json', 'cmrc2018_test.json')


# Create a DatasetDict for easier handling
train_dataset = Dataset.from_json('cmrc2018_train.json')
validation_dataset = Dataset.from_json('cmrc2018_dev.json')
test_dataset = Dataset.from_json('cmrc2018_test.json')

dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})


# Prepare for Hugging Face upload (requires authentication)
# You'll need a Hugging Face account and an access token.

# from huggingface_hub import login
# login() #  Login to Hugging Face, follow the instructions

# from huggingface_hub import upload_dataset
# upload_dataset(dataset, "your-huggingface-username/your-dataset-name", token="your-huggingface-token")


#Example of accessing data after conversion
print(dataset["train"][0])

