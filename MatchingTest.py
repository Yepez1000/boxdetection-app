import pandas as pd
import re

# Load the CSV into a DataFrame
df = pd.read_csv('/Users/edgaryepez/AmericanShip/AmericanShip2/tflite1/objdetection/CustomersView.csv')

def extract_customer_info(text):
    """Extract customer information from the provided text."""
    # Initialize variables
    first_name = ''
    last_name = ''
    as_number = None

    # Clean up text by removing non-alphanumeric characters (excluding spaces)
    cleaned_text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)

    # Split the cleaned text into words
    words = cleaned_text.split()

    # Extract first name (assuming it's the first word)
    if words:
        first_name = words[0]

    # Convert words list to string for further processing
    text_str = ' '.join(words)

    # Extract AS number using regex
    as_number_match = re.search(r'AS\d{5}', text_str)
    if as_number_match:
        as_number = int(as_number_match.group()[2:])



    # Ensure 'Unique STE # AS-' column is of string type
    df['Unique STE # AS-'] = df['Unique STE # AS-'].astype(str)

    # Query DataFrame for matching first name
    potential_matches = df[df['First Name'].str.lower() == first_name.lower()]

    # Further refine matches by last name
    for _, row in potential_matches.iterrows():
        last_name = row['Last Name']
        if re.search(last_name, text_str, re.IGNORECASE):
            return True, row

    # Query DataFrame for matching AS number
    if as_number is not None:
        as_matches = df[df['Unique STE # AS-'].str.contains(str(as_number), na=False)]

        if not as_matches.empty:
            first_name = as_matches['First Name']
            last_name = as_matches['Last Name']
            
            print(as_matches)
            # Create a regex pattern to search for the names
            pattern2 = rf"{first_name} "
            pattern3 = rf"{last_name}"

            # Check if the names are found in the text
            if re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE):
                return True, as_matches


    return False

# Example usage
text = """ DOGAN AOIYAMAN
2450 COURAGE ST STE 108- AS13264
_ BROWNSVILLE TX 78571-5133"""

print(extract_customer_info(text))  # Should return True if a match is found
