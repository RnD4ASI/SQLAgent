# create_sample_rdata.R
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table")
}
library(data.table)

sample_df <- data.table(
    id = 1:3,
    name = c("Alice", "Bob", "Charlie"),
    value = c(10.5, 20.3, 30.8),
    is_active = c(TRUE, FALSE, TRUE)
)

# Save to the UPLOAD_FOLDER if it's defined and accessible, otherwise current dir
# For the test, we'll want it in the root or UPLOAD_FOLDER.
# The Python test will look for it in app.config['UPLOAD_FOLDER'] or relative to test execution.
# Let's save it to the UPLOAD_FOLDER which the test client fixture ensures exists.
# However, UPLOAD_FOLDER is relative to app/backend, so for Rscript from root, path needs adjustment.
# For simplicity in this subtask, save to current dir, test will find it there.
save(sample_df, file = "test_r_data.Rdata")
print("test_r_data.Rdata created successfully in current directory.")
