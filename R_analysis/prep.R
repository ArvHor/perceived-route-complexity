library(dplyr)
remove_outliers_iqr <- function(data, column_name) {
  # Calculate Q1, Q3, and IQR
  Q1 <- quantile(data[[column_name]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[column_name]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  
  # Define upper and lower bounds
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  # Filter out outliers
  filtered_data <- data[data[[column_name]] >= lower_bound & data[[column_name]] <= upper_bound, ]
  
  return(filtered_data)
}
assign_misalignment_group <- function(data, distance_col) {

  data$grid_alignment_group <- "medgrid_aligned"

  data$grid_alignment_group[data$gridlike_group == "0-0.2"] <- "nogrid"

  data$grid_alignment_group[data[[distance_col]] %in% c(3, 4, 5)] <- "medgrid_misaligned"
  
  data$grid_alignment_group[
    data[[distance_col]] %in% c(3, 4, 5) & (data[["gridlike_group"]] == "0.8-1" | data[["gridlike_group"]] == "0.6-0.8")
  ] <- "highgrid_misaligned"
  
  data$grid_alignment_group[
    !(data[[distance_col]] %in% c(3, 4, 5)) & (data[["gridlike_group"]] == "0.8-1" | data[["gridlike_group"]] == "0.6-0.8")
  ] <- "highgrid_aligned"
  
  data$grid_alignment_group <- factor(data$grid_alignment_group)
  
  return(data)
}
calculate_deviation_ratio <- function(route, line) {
  #deviation_ratio <- abs(route - line) / line
  deviation_ratio <- route / line
  return(deviation_ratio)
}

shortest_higher <- function(data) {
  data$shortest_higher <- data$shortest_path_complexity < data$simplest_path_complexity
  data$shortest_higher <- factor(data$shortest_higher, levels = c(TRUE, FALSE))
  return(data)
}

assign_misalignment_group <- function(data, distance_col) {
  
  data$grid_alignment_group <- "medgrid_aligned"
  
  data$grid_alignment_group[data$gridlike_group == "0-0.2"] <- "nogrid"
  
  data$grid_alignment_group[data[[distance_col]] %in% c(3, 4, 5)] <- "medgrid_misaligned"
  
  data$grid_alignment_group[
    data[[distance_col]] %in% c(3, 4, 5) & (data[["gridlike_group"]] == "0.8-1" | data[["gridlike_group"]] == "0.6-0.8")
  ] <- "highgrid_misaligned"
  
  data$grid_alignment_group[
    !(data[[distance_col]] %in% c(3, 4, 5)) & (data[["gridlike_group"]] == "0.8-1" | data[["gridlike_group"]] == "0.6-0.8")
  ] <- "highgrid_aligned"
  
  data$grid_alignment_group <- factor(data$grid_alignment_group)
  
  return(data)
}

# Import the data and remove length outliers which have been identified using IQR method.
data <- read.csv("od_pair_data.csv")
data_raw <- read.csv("od_pair_data.csv")

data_raw$closest_strongest_lag <- factor(data_raw$closest_strongest_lag, levels = c(-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9))
print(summary(data_raw$closest_strongest_lag))
data_raw$area <- data_raw$area / 1000000
print(summary(data_raw$area))
above_30 <- data_raw[data_raw$area > 30,]
print(length(above_30$area))
columns <- colnames(data)
print(columns)
# Remove length outliers 
data <- data[data$length_outliers == "False",]
data <- data[data$complexity_outliers == "False",]


# set gridlike median groups as factors

data$gridlike_median <- factor(data$gridlike_median, levels = c("above_median","below_median"))

# fix area errors
data$area <- data$area / 1000000
data <- data[data$area < 30,]
print(summary(data$area))
# Normalize the complexity column
data <- data %>%
  mutate(shortest_path_complexity_normalized = shortest_path_complexity / max(shortest_path_complexity))

data <- data %>%
  mutate(simplest_path_complexity_normalized = simplest_path_complexity / max(shortest_path_complexity))


# Normalize the ICD measure
data <- data %>%
  mutate(intersection_density_km_normalized = intersection_density_km / max(intersection_density_km))


#data <- remove_outliers_iqr(data, "intersection_density_km")



print(summary(data$shortest_path_complexity_normalized))
print(summary(data$simplest_path_complexity_normalized))

# Check whether the complexity of any shortest routes is lower than the simplest route
data <- shortest_higher(data)
print(summary(data$shortest_higher))

# Calculate the difference between euclidean distance and simplest path distance
data$simple_straight_diff <- data$simplest_path_length - data$od_distance
data$shortest_simple_complexity_diff <- data$shortest_path_complexity - data$simplest_path_complexity

# Group the data based on if it is misaligned with a grid or not.
data <- assign_misalignment_group(data, "alignment")

# Calculate how much longer the simplest and shortest route is as a ratio.
data$shortest_diff_ratio <- calculate_deviation_ratio(data$shortest_path_length, data$od_distance)
data$simplest_diff_ratio <- calculate_deviation_ratio(data$simplest_path_length, data$od_distance)


#---- Fix the alignment measures to be treated as ordinal levels, or ordered factors ----
# 1. Closest best correlation, has levels from 0 to 8
data$alignment <- factor(data$alignment, levels = c(0,1,2,3,4,5,6,7,8),ordered = TRUE)

# 2. closest strongest correlation 
print(summary(data$closest_strongest_lag))


# 3. Strongest correlation lag
print(summary(data$strongest_correlation_lag))
data$strongest_correlation_lag <- factor(data$cross_correlation_dist, levels = c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
print(summary(data$strongest_correlation_lag))

# --- Same for cross correlation measured the normal non-circular way.---
# 1. basic has 0-17
print(summary(data$cross_correlation_dist))
data$cross_correlation_dist <- factor(data$cross_correlation_dist, levels = c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
print(summary(data$cross_correlation_dist))

# 2. Create a new column with modulo 9
dist_levels <- levels(data$cross_correlation_dist)
modulo_levels <- as.numeric(dist_levels) %% 9
data$cross_correlation_dist_9 <- factor(modulo_levels[match(data$cross_correlation_dist, dist_levels)])


# ----- Fix the factors in circular cross-correlation also ----
# 1. Full scale from -18 to 17
print(summary(data$circular_crosscorrelation_dist_weighted))
data$circular_crosscorrelation_dist_weighted <- factor(data$circular_crosscorrelation_dist_weighted, levels = c(-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
print(summary(data$circular_crosscorrelation_dist_weighted))

# 2. As absolute values
data$circular_crosscorrelation_dist_weighted_abs <- abs(as.numeric(data$circular_crosscorrelation_dist_weighted))
data$circular_crosscorrelation_dist_weighted_abs <- factor(data$circular_crosscorrelation_dist_weighted, levels = c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))

print(colnames(data))

# ----- set gridlike_groups to factors -----
data$gridlike_group <- factor(data$gridlike_group, levels = c("0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1"),ordered=TRUE)

# intersection density
data$intersection_density_km <- as.numeric(data$intersection_density_km)
print(summary(data$intersection_density_km))

# ICD
print(summary(data$k_avg))
print(summary(data$streets_per_node_avg))

data$continuous_alignment <- as.numeric(data$alignment)


# ---- remove unused data -----

columns_to_drop <- c("Unnamed..0","circular_crosscorrelation_dist_weighted","max_circular_correlation_weighted",
                     "cross_correlation_dist_weighted","simplest_path_n_nodes","shortest_path_n_nodes",
                     "shortest_path_deviation_from_prototypical","simplest_path_deviation_from_prototypical",
                     "shortest_path_instruction_equivalent","simplest_path_instruction_equivalent",
                     "shortest_path_node_degree","simplest_path_node_degree",
                     "cross_correlation_dist_9","circular_crosscorrelation_dist_weighted_abs")

data_shareable <- data[,!(names(data) %in% columns_to_drop)]

write.csv(data_shareable, "od_pair_data_public.csv", row.names = FALSE)
saveRDS(data_shareable, file = "od_pair_data_public.rds")

saveRDS(data, file = "od_pair_data_clean.rds")

library(jmvconnect)
library(jmvReadWrite)
#jmvReadWrite::write_omv(data, "od_pair_data.omv")



