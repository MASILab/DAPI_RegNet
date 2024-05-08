rm(list = ls())
library(GammaGateR)
library(gridExtra)
library(ggpubr)
library(hrbrthemes)
set.seed(42)
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Insufficient arguments provided. Usage: Rscript generate_cell_stats.R <path_to_csv_file> <output_name>", call. = FALSE)
}

csv_file <- args[1]
name <- args[2]
output_path <- "/fs5/p_masi/rudravg/MxIF_Vxm_Registered/sheets"

cell <- read.csv(csv_file)

cell$slide_id <- 1

allMarkers = grep('^Mean', names(cell), value=TRUE)
nzNormedMarkers = paste0('nzNorm_',allMarkers)
cell = do.call(rbind, lapply(split(cell, cell$slide_id), function(df){df[,nzNormedMarkers] = log10(1+sweep(df[,allMarkers], 2, colMeans(replace(df[,allMarkers], df[,allMarkers]==0, NA), na.rm = TRUE ), FUN = '/' ) ); df }) )
groupPolarisFit <- groupGammaGateR(cell[,nzNormedMarkers], 
                                   slide = cell$slide_id,
                                   n.cores = 1)

post_all <- do.call(rbind, (lapply(groupPolarisFit, "[[", "expressionW")))
write.csv(post_all, file.path(output_path, paste0(name, "_post_all.csv")), row.names = FALSE)


