#################################################
# R Script to Convert Excel Files to CSV Format #
#################################################

install.packages('readxl')
install.packages('readr')
library(readxl)
library(readr)

MLS_Land_Sales <- read_excel("data/2022 - 2025 MLS Land Sales.xlsm")
head(MLS_Land_Sales)

write_csv(MLS_Land_Sales, "converted_MLS_Land_Sales.csv")

Multi_Family_MLS_Export <- read_excel("data/2022 - 2025 Multi-Family MLS Export.xlsm")
head(Multi_Family_MLS_Export)

write_csv(Multi_Family_MLS_Export, "converted_Multi_Family_MLS_Export.csv")

MLS_Preped_Sales_for_PUMA <- read_excel("data/2025 MLS Prepared Sales For PUMA Upload.xlsx")
head(MLS_Preped_Sales_for_PUMA)

write_csv(MLS_Preped_Sales_for_PUMA, "converted_MLS_Preped_Sales_for_PUMA.csv")

MLS_Residential_Sales <- read_excel("data/2025 MLS Residential Sales.xlsm")
head(MLS_Residential_Sales)

write_csv(MLS_Residential_Sales, "converted_MLS_Residential_Sales.csv")

Parcel_Export <- read_excel("data/Parcel Export.xlsx")
head(Parcel_Export)

write_csv(Parcel_Export, "converted_Parcel_Export.csv")
