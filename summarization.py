from transformers import pipeline

def summarize_text_file(file_path, max_length=130, min_length=30, do_sample=False):
    # Initialize the summarizer
    summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
    
    # Read the contents of the file
    with open(file_path, 'r') as file:
        text = file.read()

    # Tokenize the text to get the tokens
    tokens = summarizer.tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]

    # Split the tokens into chunks of max_length
    chunks = [tokens[i:i + 1024] for i in range(0, len(tokens), 1024)]

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        # Convert the tokens back to text
        chunk_text = summarizer.tokenizer.decode(chunk)
        
        # Print the length of the chunk
        print(f"Length of chunk {i}: {len(chunk_text)}")

        # Check if the chunk_text is not empty
        if chunk_text.strip():
            try:
                # Summarize the chunk text
                summary = summarizer(chunk_text, max_length=max_length, min_length=min_length, do_sample=do_sample)
                # Add the summary to the list of summaries
                summaries.append(summary)
            except IndexError as e:
                print(f"Error while processing chunk {i}: {e}")
                print(f"Contents of chunk {i}: {chunk_text}")

    return summaries

print(summarize_text_file('converttest_transcription.txt'))
