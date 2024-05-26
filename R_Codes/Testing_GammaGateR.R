rm(list = ls())
set.seed(42)

library(GammaGateR)
library(gridExtra)
library(ggpubr)
library(hrbrthemes)

csv_file <- '/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/GCA033TIB_TISSUE01/combined_GCA033TIB_instances.csv'
name <- 'GCA033TIB_TISSUE01'
output_path <- "/fs5/p_masi/rudravg/MxIF_Vxm_Registered/metrics"


cell <- read.csv(csv_file)

allMarkers = grep('^Mean', names(cell), value=TRUE)
nzNormedMarkers = paste0('nzNorm_',allMarkers)

cell = do.call(rbind, lapply(split(cell, cell$slide_id), function(df){df[,nzNormedMarkers] = log10(1+sweep(df[,allMarkers], 2, colMeans(replace(df[,allMarkers], df[,allMarkers]==0, NA), na.rm = TRUE ), FUN = '/' ) ); df }) )


for(i in 1:length(allMarkers)){
  cat('\n### ', allMarkers[i], " \n")
  col.name <- nzNormedMarkers[i]
  slidetest <- cell[, c("slide_id", col.name)]
  slidetest <- slidetest[slidetest[,col.name]!=0,]
  colnames(slidetest)[2] <- "mkr.val"
  p1 <- ggplot() + ggtitle(col.name)+
    geom_freqpoly(data=slidetest, aes(mkr.val,y=after_stat(density), color=slide_id,group=slide_id), binwidth = 0.05, alpha=0.1, linewidth=1)+theme_ipsum()+theme(legend.position = "none")
  print(p1)
  cat('\n\n')
}


boundaries = list(NULL, #CD11B
                  NULL, #CD29
                  NULL, #CD3d
                  NULL, #CD45
                  NULL, #CD4
                  NULL, #CD68
                  matrix(c(0,.25, .3, Inf),byrow = TRUE, nrow=2), #CD8
                  NULL, #CgA
                  NULL, #Lysozome
                  NULL, #NaKATPase
                  NULL, #PanCK
                  NULL, #SMA
                  NULL, #Sox9
                  NULL, #Vimentin
                  NULL) #OLFM4

groupPolarisFit <- groupGammaGateR(cell[,nzNormedMarkers], 
                                   slide = cell$slide_id,
                                   n.cores = 1)


for (i in (nzNormedMarkers)){
  mkr.name <- strsplit(i, split="_")[[1]][3]
  cat('\n### ', mkr.name, " \n")
  temp <- groupPolarisFit[1:2]
  class(temp) <- "groupGammaGateR"
  plotss <- plot(temp, marker=i, diagnostic=FALSE, histogram=TRUE, print=FALSE, tabl=TRUE)
  print(do.call(ggarrange,plotss))
  class(temp) <- "groupGammaGateR"
  cat('\n\n')
}

post_all <- do.call(rbind, (lapply(groupPolarisFit, "[[", "expressionZ")))
write.csv(post_all, file.path(output_path, paste0(name, "_post_all_V2.csv")), row.names = FALSE)