import requests
import time

def calculate_accuracy_and_response_time(test_cases, url):
    correct_count = 0
    total_response_time = 0

    for case in test_cases:
        symptom = case['symptom']
        correct_drugs = set(case['correct_drugs'])

        # Measure response time
        payload = {"symptom": symptom}
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()

        # Calculate response time
        response_time = end_time - start_time
        total_response_time += response_time

        # Get recommended drugs from response
        recommended_drugs = set(response.json().get('recommendations', []))

        # Calculate accuracy for this test case
        common_drugs = correct_drugs.intersection(recommended_drugs)
        if len(common_drugs) > 0:
            correct_count += 1
    
    # Calculate average accuracy and response time
    accuracy = correct_count / len(test_cases) * 100
    average_response_time = total_response_time / len(test_cases)
    return accuracy, average_response_time

# Example test cases
test_cases = [
    {"symptom": "headache", "correct_drugs": ["Tylenol", "Ibuprofen", "Excedrin"]},
    {"symptom": "anxiety", "correct_drugs": ["Xanax", "Lexapro", "Zoloft"]},
    {"symptom": "depression", "correct_drugs": ["Prozac", "Zoloft", "Lexapro"]},
    {"symptom": "hypertension", "correct_drugs": ["Lisinopril", "Amlodipine", "Metoprolol"]},
    {"symptom": "diabetes", "correct_drugs": ["Metformin", "Januvia", "Glipizide"]},
    {"symptom": "insomnia", "correct_drugs": ["Ambien", "Lunesta", "Trazodone"]},
    {"symptom": "acne", "correct_drugs": ["Accutane", "Doxycycline", "Clindamycin"]},
    {"symptom": "allergy", "correct_drugs": ["Claritin", "Zyrtec", "Allegra"]},
    {"symptom": "asthma", "correct_drugs": ["Albuterol", "Singulair", "Advair"]},
    {"symptom": "pain", "correct_drugs": ["Ibuprofen", "Acetaminophen", "Oxycodone"]}
]


if __name__ == "__main__":
    url = "http://127.0.0.1:5000/predict"
    accuracy, avg_response_time = calculate_accuracy_and_response_time(test_cases, url)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Response Time: {avg_response_time:.4f} seconds")
