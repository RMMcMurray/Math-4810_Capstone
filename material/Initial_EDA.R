##################################
# Initial EDA on Land Sales Data #
##################################

install.packages("tidyverse")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("MASS")
install.packages("GGally")
install.packages("readxl")
library(tidyverse)
library(dplyr)
library(ggplot2)
library(MASS)
library(GGally)
library(readxl)

land_sales <- read_excel("data/2022 - 2025 MLS Land Sales.xlsm")
head(land_sales)
view(land_sales)

ggplot(land_sales, aes(x = `Original List Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Original List Prices", x = "Original List Price", y = "Frequency")

ggplot(land_sales, aes(x = `List Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Land Sale Prices", x = "Sale Price", y = "Frequency")

ggplot(land_sales, aes(x = `Sold Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Sold Prices", x = "Sold Price", y = "Frequency")

mean(land_sales$`Original List Price` - land_sales$`List Price`, na.rm = TRUE)
count(land_sales |> filter(`Original List Price` > `List Price`))

mean(land_sales$`List Price` - land_sales$`Sold Price`, na.rm = TRUE)
count(land_sales |> filter(`List Price` > `Sold Price`))

ggpairs(land_sales[, c("Sold Price", "Total SqFt", "Basement SqFt", "Main SqFt", 
"Upstairs SqFt", "Lot Acres", "Total Bedrooms", "Full Baths", "Half Baths", "Days on Market")])

#########################################
# Initial EDA on Residential Sales Data #
#########################################

residential_sales <- read_excel("data/2025 MLS Residential Sales.xlsm")
head(residential_sales)
view(residential_sales)

ggplot(residential_sales, aes(x = `Original List Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Original List Prices", x = "Original List Price", y = "Frequency")

ggplot(residential_sales, aes(x = `List Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of List Prices", x = "Sale Price", y = "Frequency")

ggplot(residential_sales, aes(x = `Sold Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Sold Prices", x = "Sold Price", y = "Frequency")

mean(residential_sales$`Original List Price` - residential_sales$`List Price`, na.rm = TRUE)
count(residential_sales |> filter(`Original List Price` > `List Price`))

mean(residential_sales$`List Price` - residential_sales$`Sold Price`, na.rm = TRUE)
count(residential_sales |> filter(`List Price` > `Sold Price`))

ggpairs(residential_sales[, c("Sold Price", "Total SqFt", "Basement SqFt", "Main SqFt", 
"Upstairs SqFt", "Lot Acres", "Total Bedrooms", "Full Baths", "Half Baths", "Days on Market")])

##########################################
# Initial EDA on Multi-Family Sales Data #
##########################################

multi_family_sales <- read_excel("data/2022 - 2025 Multi-Family MLS Export.xlsm")
head(multi_family_sales)
view(multi_family_sales)

ggplot(multi_family_sales, aes(x = `Original List Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Original List Prices", x = "Original List Price", y = "Frequency")

ggplot(multi_family_sales, aes(x = `List Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of List Prices", x = "Sale Price", y = "Frequency")

ggplot(multi_family_sales, aes(x = `Sold Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Sold Prices", x = "Sold Price", y = "Frequency")

mean(multi_family_sales$`Original List Price` - multi_family_sales$`List Price`, na.rm = TRUE)
count(multi_family_sales |> filter(`Original List Price` > `List Price`))

mean(multi_family_sales$`List Price` - multi_family_sales$`Sold Price`, na.rm = TRUE)
count(multi_family_sales |> filter(`List Price` > `Sold Price`))

ggpairs(multi_family_sales[, c("Sold Price", "Lot Acres", "Total Units", "Covered Parking", 
"Finished SqFt")])

##################################
# Initial EDA on PUMA Sales Data #
##################################

puma_sales <- read_excel("data/2025 MLS Prepared Sales For PUMA Upload.xlsx")
head(puma_sales)
view(puma_sales)

ggplot(puma_sales, aes(x = `List Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of List Prices", x = "Sale Price", y = "Frequency")

ggplot(puma_sales, aes(x = `Sold Price`)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Sold Prices", x = "Sold Price", y = "Frequency")

mean(puma_sales$`List Price` - puma_sales$`Sold Price`, na.rm = TRUE)
count(puma_sales |> filter(`List Price` > `Sold Price`))

ggpairs(puma_sales[, c("Sold Price", "Acres", "Basement Finished", "Basement Square Feet", 
"Total Bedrooms", "Total Full Bathrooms", "Total Half Bathrooms", "Total Three-quarter Bathrooms", 
"Total Square Feet", "Total Fireplaces", "Total Kitchens", "Total Laundry Rooms")])
