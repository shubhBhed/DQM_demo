import pandas

#csv_path = "./data/folder_with_data/yellow_tripdata_sample_2019-01.csv"
dataframe = pandas.read_csv("v_dqm_kpi.csv")

batch_parameters = {"dataframe": dataframe}

import great_expectations as gx

context = gx.get_context()
#set_up_context_for_example(context)

# Retrieve a Validation Definition that uses the dataframe Batch Definition
validation_definition_name = "my_validation_definition"
validation_definition = context.validation_definitions.get(validation_definition_name)

# Validate the dataframe by passing it to the Validation Definition as Batch Parameters.
validation_results = validation_definition.run(batch_parameters=batch_parameters)
print(validation_results)