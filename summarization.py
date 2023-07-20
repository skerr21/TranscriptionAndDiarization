import os
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
    chunk_summaries = []
    for chunk in chunks:
        # Convert the tokens back to text
        chunk_text = summarizer.tokenizer.decode(chunk)
        
        # Check if the chunk_text is not empty
        if chunk_text.strip():
            try:
                # Summarize the chunk text
                summary = summarizer(chunk_text, max_length=max_length, min_length=min_length, do_sample=do_sample)
                # Add the summary to the list of summaries
                chunk_summaries.append(summary[0]['summary_text'])
            except IndexError as e:
                print(f"Error while processing chunk: {e}")
                print(f"Contents of chunk: {chunk_text}")

    # Create a single string from all chunk summaries
    all_summaries_text = ' '.join(chunk_summaries)
    print("All Summaries Text: " + all_summaries_text)
    # Create an overall summary of the chunk summaries
    overall_summary = summarizer(all_summaries_text, max_length=max_length, min_length=min_length, do_sample=do_sample)

    return overall_summary


print(summarize_text_file('hourtest_enhanced_transcription.txt'))