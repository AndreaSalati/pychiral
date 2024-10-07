source("/Users/salati/Documents/CODE/github/pyCHIRAL/CHIRAL.R")  

# Import the CSV file
data <- read.csv("/Users/salati/Documents/CODE/github/pyCHIRAL/s_log.csv", row.names = 1)  # Assuming first column is gene names and data is samples × genes
ZT  <- ZT <- c(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 18, 18, 20, 20, 22, 22)

# ZT <- as.numeric(as.character(ZT))
# Ensure the data is in the right format
# Data should have genes on rows and samples on columns
E <- as.matrix(data)  # Convert data frame to matrix if necessary
ccg <- c(
  "Arntl",
  "Npas2",
  "Cry1",
  "Cry2",
  "Per1",
  "Per2",
  "Nr1d1",
  "Nr1d2",
  "Tef",
  "Dbp",
  "Ciart",
  "Per3",
  "Bmal1"
)
debug(CHIRAL)
# Call the CHIRAL function
result <- CHIRAL(E = E, iterations = 500, clockgenes = ccg, tau2 = NULL, u = NULL, 
                 sigma2 = NULL, TSM = TRUE, mean.centre.E = TRUE, 
                 q = 0.1, update.q = FALSE, pbar = TRUE, phi.start = NULL, 
                 standardize = FALSE, GTEx_names = FALSE)

# Print or inspect the result
print(result)

true_phi <- ZT * 2 * pi / 24 
new_phi <- result$phi

plot(true_phi, new_phi, xlab = "True Phi", ylab = "New Phi", main = "True Phi vs New Phi", pch = 16, col = "blue")



# Optionally, save the result to an RDS file for later use
# saveRDS(result, "chiral_result.rds")