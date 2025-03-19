import great_expectations as gx


def test_dataset(df, context):
    """Test dataset quality and integrity."""

    # Create an Expectation Suite
    suite_name = "sentiment_expectation_suite"
    suite = gx.ExpectationSuite(name=suite_name)

    # Add the Expectation Suite to the Data Context
    suite = context.suites.add(suite)

    # Connect to data and create a Batch.
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "batch definition"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Create expectaions
    column_sentiment_set = ["negative", "neutral", "positive"]

    column_name = list(df.columns)
    distinct_expectation = gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="sentiment", value_set=column_sentiment_set
    )  # expected labels
    compound_col_expectation = gx.expectations.ExpectCompoundColumnsToBeUnique(
        column_list=["text", "id"]
    )  # data leaks
    null_expectation = gx.expectations.ExpectColumnValuesToNotBeNull(
        column="sentiment", mostly=0.9
    )  # missing values
    headline_type_expectation = gx.expectations.ExpectColumnValuesToBeOfType(
        column="text", type_="str"
    )  # type adherence
    description_type_expectation = (
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="sentiment", type_="str"
        )
    )  # type adherence
    available_col_expectation = (
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=column_name
        )
    )  # schema adherence
    suite.add_expectation(distinct_expectation)
    suite.add_expectation(compound_col_expectation)
    suite.add_expectation(null_expectation)
    suite.add_expectation(headline_type_expectation)
    suite.add_expectation(description_type_expectation)
    suite.add_expectation(available_col_expectation)

    # Validate dataset
    validation_result = batch.validate(suite)
    assert validation_result["success"]
