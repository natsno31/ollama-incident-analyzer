import pandas as pd
import ollama


def call_local_ollama_llm(prompt: str, model: str) -> str:
    """
    Calls a local Ollama Language Model (LM) and returns its response.

    Parameters:
    prompt (str): The prompt to be submitted to the local Ollama LM.
    model (str): The name of the LM to be used.

    Returns:
    str: A string containing the response from the local Ollama LM called.
    """
    
    # generate the Ollama response
    response: object = ollama.generate(model = model, prompt = prompt, stream=False)

    # turn the response into a string
    answer: str = response['response']

    return answer


def q1_get_canada_incidents(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an new DataFrame containing only the rows where the value of the country column is Canada. In addition, 
    the DataFrame only contains the columns number, vendor, and country.

    Parameters:
    original_df (pd.DataFrame): The original DataFrame read from the disk.

    Returns:
    pd.DataFrame: A new DataFrame containing only the rows with Canada and the number, vendor, and country columns.
    """

    # collect only the rows where the country is Canada
    new_df: pd.DataFrame = get_canada(original_df)

    # get only number, vendor, and country columns
    new_df: pd.DataFrame = new_df.drop(['incident_state', 'knowledge', 'product', 'summary'], axis=1)

    return new_df


def get_canada(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a new Dataframe containing only the rows where the value of the country is Canada.

    Parameters:
    original_df (pd.DataFrame): The original DataFrame sent to the function for filtering.

    Returns:
    pd.DataFrame: A new DataFrame containing only the rows where the value of the country column is Canada.
    """
    
    # collect only the rows where the country is Canada
    new_df: pd.DataFrame = original_df[original_df['country'] == 'Canada']

    return new_df


def q2_get_canada_awaiting_info_incidents(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an new DataFrame containing only the rows where the value of the country column is Canada 
    and the value of the incident_state column is Awaiting User Info. In addition, the DataFrame contains only
    the columns number, knowledge, product, and country.

    Parameters:
    original_df (pd.DataFrame): The original DataFrame read from the disk.

    Returns:
    pd.DataFrame: A new DataFrame with only rows containing Canada and Awaiting User Info and only number, knowledge, 
    product, and country columns.
    """
    
    # collect only the rows where the incident state is Awaiting User Infor
    new_df: pd.DataFrame = original_df[original_df['incident_state'] == 'Awaiting User Info']
    
    # collect only the rows where the country is Canada from the filtered DataFrame
    new_df: pd.DataFrame = get_canada(new_df)

    # get only the number, knowledge, product, and country columns
    new_df: pd.DataFrame = new_df.drop(['incident_state', 'vendor', 'summary'], axis=1)

    return new_df


def q3_get_num_days_between_incident_opening_closure(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a new DataFrame with only number and delta columns containing the incident number and the
    difference between the opening and closing days.

    Parameters:
    original_df (pd.DataFrame): The original DataFrame read from the disk.

    Returns:
    pd.DataFrame: a new DataFrame containing only the incident numbers and the differences between the opening and closing days.
    """
    
    model: str = "phi4-mini:latest" # model for Ollama
    incident_nums: list[str] = original_df['number'].tolist() # the incident numbers from the DataFrame
    summaries: list[str] = original_df['summary'].tolist() # the summaries from the DataFrame
    num_days: list[int] = [] # list for the number of days between opening and closing

    # iterrate through the summaries and send them to Ollama to get the difference between the opening and closing days
    for i in summaries:
        
        # Ollama prompt for extracting the opening day
        prompt: str = (
            f"Extract the date that directly follows 'opened on' from the following text.{i}"
            f"If the text uses the format ##/##/####, ##/#/####, or #/##/#### for the date, then interpret it as DD/MM/YYYY."
            f"Ignore mentions of today and only look for the date strictly following 'opened on'."
            f"Output only the 'opened on' date as numbers in a string strictly formatted YYYY-MM-DD with no quotation marks." 
            f"Do not use any quotation marks and no other words in the output."
            f"Make sure that the output has the year (YYYY) first, month (MM) second, and the day (DD) last."
            f"Do not provide any explaination."
        )

        # get the opening day
        open_date: str = no_quotations(call_local_ollama_llm(prompt, model))
        
        # Ollama prompt for extracting the closing day
        prompt: str = (
            f"Extract the date that directly follows 'closed on' from the following text.{i}"
            f"If the text uses the format ##/##/####, ##/#/####, or #/##/#### for the date, then interpret it as DD/MM/YYYY."
            f"Ignore mentions of today and only look for the date strictly following 'closed on'."
            f"Output only the 'closed on' date as numbers in a string strictly formatted YYYY-MM-DD with no quotation marks." 
            f"Do not use any quotation marks and no other words in the output."
            f"Make sure that the output has the year (YYYY) first, month (MM) second, and the day (DD) last."
            f"Do not provide any explaination."
        )

        # get the closing day
        close_date: str = no_quotations(call_local_ollama_llm(prompt, model))

        # add the number of days difference to the num_days list
        num_days.append(get_difference(open_date, close_date))

    # create a new dictionary with the two columns for the new DataFrame
    data: dict[str, list] = {'number': incident_nums, 'delta': num_days}

    # get the new DataFrame using the new dictionary
    new_df: pd.DataFrame = pd.DataFrame(data)

    return new_df


def get_difference(open_date: str, close_date: str) -> int:
    """
    Gets the difference between the opening and closing dates and returns the days.

    Parameters:
    open_date (str): The original string for the opening date from Ollama.
    close_date (str): The original string for the closing date from Ollama.

    Returns:
    int: The number of days between the opening and closing date.
    """
    # convert the opening and closing days to Timestamps
    open: pd.Timestamp = pd.to_datetime(open_date)
    close: pd.Timestamp = pd.to_datetime(close_date)

    # get the difference between the opening and closing days
    difference: pd.Timedelta = close - open

    return difference.days


def no_quotations(output: str) -> str:
    """
    Removes extra quotation marks around a string if needed and returns the string.

    Parameters:
    output (str): The original string generated from Ollama to check if there are extra quotation marks.

    Returns:
    str: A string containing the original value, but with no extra quotations.
    """

    # check if the string had extra quotation marks
    if output.startswith('"') and output.endswith('"'):
        return output[1:-1] # return without the quotation marks

    # if slicing isn't needed, then just return the original string
    return output


def q4_get_incident_urgency(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a new DataFrame with only number and urgency columns containing the incident number and the
    level of urgency.

    Parameters:
    original_df (pd.DataFrame): The original DataFrame read from the disk.

    Returns:
    pd.DataFrame: a new DataFrame containing only the incident number and urgency.
    """
    
    model: str = "phi4-mini:latest" # model for Ollama
    incident_nums: list[str] = [] # list for the incident numbers
    summaries: list[str] = original_df['summary'].tolist() # list of the summaries from the DataFrame
    urgency: list[str] = [] # list for the urgency levels

    # iterrate through the summaries to get the incident numbers and urgency levels from Ollama
    for i in summaries:
        
        # Ollama prompt for the incident number
        prompt: str = (
            f"Extract the number starting with 'INC' after 'Incident' in the following text and include the INC in front. {i}"
            f"Output only the number in a string strictly formatted INC# with no quotation marks." 
            f"Do not use any quotation marks and no other words in the output."
        )

        # add the incident number to the list
        incident_nums.append(call_local_ollama_llm(prompt, model))

        # Ollama prompt for the urgency level
        prompt: str = (
            f"Extract if the level directly before 'impact and urgency' is 'moderate', 'medium' or 'high' and not more than one, in the following text. {i}"
            f"Output only the extracted level with the first letter capitalized in a string strictly formatted as 'Moderate', 'Medium', or 'High' and nothing else with no quotation marks." 
            f"Do not use any quotation marks and no other words in the output."
        )

        # add the urgency level to the list
        urgency.append(call_local_ollama_llm(prompt, model).capitalize())

    # create a dictionary for the new DataFrame
    data: dict[str, list[str]] = {'number': incident_nums, 'urgency': urgency}
    
    # get the new DataFram using the dictionary
    new_df: pd.DataFrame = pd.DataFrame(data)

    return new_df

