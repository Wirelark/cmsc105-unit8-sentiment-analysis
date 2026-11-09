
# Name: John Robin
# Course: CMSC 105 6902 - Introduction to Programming in Python
# Project: Unit 8 - Hugging Face Sentiment Analysis
# Description: This program uses a pre-trained Hugging Face AI model
# to analyze movie reviews. The user can enter multiple reviews,
# and the program counts positive and negative results and displays
# a final report with the total reviews and positive percentage.


def load_sentiment_model():
    """
    Load the pre-trained Hugging Face sentiment analysis model.

    Parameters:
        None

    Returns:
        object: The pre-trained sentiment analysis model.
    """
    # Import the pipeline function to load the AI model.
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )


def load_reviews():
    """
    Collect movie reviews from the user until q is entered.

    Parameters:
        None

    Returns:
        list: A list containing the movie reviews entered by the user.
    """
    reviews = []

    # Continue asking for reviews until the user enters q.
    while True:
        print("Enter next review, or q to quit.")
        review = input()

        if review == "q":
            break

        reviews.append(review)

    return reviews


def get_sentiment(sentiment_model, review):
    """
    Analyze one movie review and return its sentiment label.

    Parameters:
        sentiment_model (object): The pre-trained AI sentiment model.
        review (str): The movie review to analyze.

    Returns:
        str: The sentiment label, POSITIVE or NEGATIVE.
    """
    # The model returns a list containing a dictionary.
    result = sentiment_model(review)

    # Retrieve the sentiment label from the first dictionary.
    sentiment = result[0]["label"]

    return sentiment


def print_sentiment_counts(sentiment_model, reviews):
    """
    Count and print the number of positive and negative reviews.

    Parameters:
        sentiment_model (object): The pre-trained AI sentiment model.
        reviews (list): The list of movie reviews to analyze.

    Returns:
        None: Prints the sentiment counts without returning a value.
    """
    positive_count = 0
    negative_count = 0

    # Analyze each review and update the correct count.
    for review in reviews:
        sentiment = get_sentiment(sentiment_model, review)

        if sentiment == "POSITIVE":
            positive_count += 1
        elif sentiment == "NEGATIVE":
            negative_count += 1

    print(f"Positive: {positive_count}")
    print(f"Negative: {negative_count}")


def get_sentiment_counts(sentiment_model, reviews):
    """
    Count the sentiments and return the results in a dictionary.

    Parameters:
        sentiment_model (object): The pre-trained AI sentiment model.
        reviews (list): The list of movie reviews to analyze.

    Returns:
        dict: A dictionary containing POSITIVE and NEGATIVE counts.
    """
    sentiment_counts = {
        "POSITIVE": 0,
        "NEGATIVE": 0
    }

    # Analyze each review and increase its sentiment count.
    for review in reviews:
        sentiment = get_sentiment(sentiment_model, review)

        if sentiment == "POSITIVE":
            sentiment_counts["POSITIVE"] += 1
        elif sentiment == "NEGATIVE":
            sentiment_counts["NEGATIVE"] += 1

    return sentiment_counts


def display_report(sentiment_dictionary):
    """
    Display a report showing sentiment counts and percentages.

    Parameters:
        sentiment_dictionary (dict): A dictionary containing the
            number of positive and negative movie reviews.

    Returns:
        None: Prints the final report without returning a value.
    """
    positive_count = sentiment_dictionary["POSITIVE"]
    negative_count = sentiment_dictionary["NEGATIVE"]

    # Calculate the total number of reviews.
    total_reviews = positive_count + negative_count

    # Avoid dividing by zero if no reviews were entered.
    if total_reviews > 0:
        positive_percentage = (positive_count / total_reviews) * 100
    else:
        positive_percentage = 0.0

    # Display the final report with two decimal places.
    print("--- FINAL REPORT ---")
    print(f"Total Reviews: {total_reviews}")
    print(f"Positive Reviews: {positive_count}")
    print(f"Negative Reviews: {negative_count}")
    print(f"Positive Percentage: {positive_percentage:.2f}%")


def main():
    """
    Run the complete movie review sentiment analysis program.

    Parameters:
        None

    Returns:
        None: Runs the program and displays the results.
    """
    # Load the pre-trained sentiment analysis model.
    sentiment_model = load_sentiment_model()

    # Collect movie reviews from the user.
    review_list = load_reviews()

    # Display the reviews entered by the user.
    print("REVIEWS:")
    for review in review_list:
        print(review)

    # Analyze the reviews and store the sentiment counts.
    sentiment_dictionary = get_sentiment_counts(
        sentiment_model, review_list
    )

    # Display the final sentiment analysis report.
    display_report(sentiment_dictionary)


if __name__ == "__main__":
    main()
