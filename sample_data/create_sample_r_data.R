# Create a sample dataframe
sample_df <- data.frame(
  ID = 1:10,
  Name = paste0("Name", 1:10),
  Value = rnorm(10)
)

# Save the dataframe to an .Rdata file
save(sample_df, file = "sample_data/sample_data.Rdata")

# Print a message to confirm completion
print("Sample data frame created and saved to sample_data/sample_data.Rdata")
