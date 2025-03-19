from tqdm import tqdm
from dotenv import load_dotenv
from langfuse import Langfuse
import pandas as pd


# Config
# Remember to specify Langfuse config in `.env` file
load_dotenv()
LF_DATASET_NAME = "eval-dataset"
LF_DATASET_DESC = "Evaluate app's responses to user questions."


def create_langfuse_dataset(testset_name) -> None:
    """
    Creates a Dataset in langfuse and populates it with example
    pairs of input questions and expected output.
    """
    # Init langfuse client and verify connection
    langfuse = Langfuse()
    langfuse.auth_check()

    # Create dataset
    langfuse.create_dataset(name=LF_DATASET_NAME, description=LF_DATASET_DESC)

    # Get dataset content
    #df = _get_dataset_content_as_dataframe()
    df = pd.read_csv(testset_name, sep=",")

    # Populate dataset with eval questions
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        langfuse.create_dataset_item(
            dataset_name=LF_DATASET_NAME,
            input=row["user_input"],
            expected_output=row["reference"],
            metadata=row["reference_contexts"],
        )


if __name__ == "__main__":
    print(f"Creating dataset: {LF_DATASET_NAME} ...")
    create_langfuse_dataset("testset.csv") #Replace <testset.csv> with the actual path to your file.
    print("Done.")