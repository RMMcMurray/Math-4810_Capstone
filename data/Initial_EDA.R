##################################
# Initial EDA on Land Sales Data #
##################################

install.packages("tidyverse")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("MASS")
install.packages("GGally")
library(tidyverse)
library(dplyr)
library(ggplot2)
library(MASS)
library(GGally)

land_sales <- read.csv("data/converted_MLS_Land_Sales.csv")
head(land_sales)

ggplot(land_sales, aes(x = Original.List.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Original List Prices", x = "Original List Price", y = "Frequency")

ggplot(land_sales, aes(x = List.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Land Sale Prices", x = "Sale Price", y = "Frequency")

ggplot(land_sales, aes(x = Sold.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Sold Prices", x = "Sold Price", y = "Frequency")

mean(land_sales$Original.List.Price - land_sales$List.Price, na.rm = TRUE)
count(land_sales |> filter(Original.List.Price > List.Price))

mean(land_sales$List.Price - land_sales$Sold.Price, na.rm = TRUE)
count(land_sales |> filter(List.Price > Sold.Price))

ggpairs(land_sales[, c("Sold.Price", "Total.SqFt", "Basement.SqFt", "Main.SqFt", 
"Upstairs.SqFt", "Lot.Acres", "Total.Bedrooms", "Full.Baths", "Half.Baths", "Days.on.Market")])

#########################################
# Initial EDA on Residential Sales Data #
#########################################

residential_sales <- read.csv("data/converted_MLS_Residential_Sales.csv")
head(residential_sales)

ggplot(residential_sales, aes(x = Original.List.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Original List Prices", x = "Original List Price", y = "Frequency")

ggplot(residential_sales, aes(x = List.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Land Sale Prices", x = "Sale Price", y = "Frequency")

ggplot(residential_sales, aes(x = Sold.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Sold Prices", x = "Sold Price", y = "Frequency")

mean(residential_sales$Original.List.Price - residential_sales$List.Price, na.rm = TRUE)
count(residential_sales |> filter(Original.List.Price > List.Price))

mean(residential_sales$List.Price - residential_sales$Sold.Price, na.rm = TRUE)
count(residential_sales |> filter(List.Price > Sold.Price))

ggpairs(residential_sales[, c("Sold.Price", "Total.SqFt", "Basement.SqFt", "Main.SqFt", 
"Upstairs.SqFt", "Lot.Acres", "Total.Bedrooms", "Full.Baths", "Half.Baths", "Days.on.Market")])

##########################################
# Initial EDA on Multi-Family Sales Data #
##########################################

multi_family_sales <- read.csv("data/converted_Multi_Family_MLS_Export.csv")
head(multi_family_sales)

ggplot(multi_family_sales, aes(x = Original.List.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Original List Prices", x = "Original List Price", y = "Frequency")

ggplot(multi_family_sales, aes(x = List.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Land Sale Prices", x = "Sale Price", y = "Frequency")

ggplot(multi_family_sales, aes(x = Sold.Price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "black") +
  labs(title = "Distribution of Sold Prices", x = "Sold Price", y = "Frequency")

mean(multi_family_sales$Original.List.Price - multi_family_sales$List.Price, na.rm = TRUE)
count(multi_family_sales |> filter(Original.List.Price > List.Price))

mean(multi_family_sales$List.Price - multi_family_sales$Sold.Price, na.rm = TRUE)
count(multi_family_sales |> filter(List.Price > Sold.Price))

ggpairs(multi_family_sales[, c("Sold.Price", "Lot.Acres", "Total.Units", "Covered.Parking", 
"Finished.SqFt")])
