## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----message = FALSE, warning = FALSE-----------------------------------------
library(VectraPolarisData)
spe_lung <- HumanLungCancerV3()
spe_ovarian <- HumanOvarianCancerVP()

## ----eval = FALSE-------------------------------------------------------------
#  library(ExperimentHub)
#  eh <- ExperimentHub()        # initialize hub instance
#  q <- query(eh, "VectraPolarisData") # retrieve 'VectraPolarisData' records
#  id <- q$ah_id[1]             # specify dataset ID to load
#  spe <- eh[[id]]              # load specified dataset

## ----echo=FALSE---------------------------------------------------------------
# image from Righelli et al. 2021
url <- "SPE.png"

## ----message = FALSE----------------------------------------------------------
library(dplyr)

## Assays slots
assays_slot <- assays(spe_ovarian)
intensities_df <- assays_slot$intensities
rownames(intensities_df) <- paste0("total_", rownames(intensities_df))
nucleus_intensities_df<- assays_slot$nucleus_intensities
rownames(nucleus_intensities_df) <- paste0("nucleus_", rownames(nucleus_intensities_df))
membrane_intensities_df<- assays_slot$membrane_intensities
rownames(membrane_intensities_df) <- paste0("membrane_", rownames(membrane_intensities_df))

# colData and spatialData
colData_df <- colData(spe_ovarian)
spatialCoords_df <- spatialCoords(spe_ovarian)

# clinical data
patient_level_df <- metadata(spe_ovarian)$clinical_data

cell_level_df <- as.data.frame(cbind(colData_df, 
                                     spatialCoords_df,
                                     t(intensities_df),
                                     t(nucleus_intensities_df),
                                     t(membrane_intensities_df))
)


ovarian_df <- full_join(patient_level_df, cell_level_df, by = "sample_id")


## -----------------------------------------------------------------------------
sessionInfo()
