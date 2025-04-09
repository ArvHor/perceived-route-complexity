data <- readRDS("od_pair_data_clean.rds")
library(ggplot2)
library(dplyr)
## ----- Shortest path ----------------

# Comparing length ratio 
plot <- ggplot(data, aes(x = alignment, y = shortest_diff_ratio, fill = gridlike_group)) +
  geom_boxplot() +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationship between alignment and the circuity of the shortest route in increasingly grid-like cities.",
       x = "Alignment",
       y = "Circuity")+guides(fill = "none")+
      theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())

ggsave(filename = "figures/shortest_circuity_grid_boxplots.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 10, 
       height = 6, 
       units = "in", 
       dpi = 600)

show(plot)

subset_data <- subset(data, gridlike_group == "0.8-1")
anova_result <- aov(shortest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)

subset_data <- subset(data, gridlike_group == "0.6-0.8")
anova_result <- aov(shortest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)
subset_data <- subset(data, gridlike_group == "0.4-0.6")
anova_result <- aov(shortest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)
subset_data <- subset(data, gridlike_group == "0.2-0.4")
anova_result <- aov(shortest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)
subset_data <- subset(data, gridlike_group == "0-0.2")
anova_result <- aov(shortest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)

# Comparing complexity
plot <- ggplot(data, aes(x = alignment, y = shortest_path_complexity_normalized, fill = gridlike_group)) +
  geom_boxplot() +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(
       x = "Alignment",
       y = "Complexity.") +
  guides(fill = "none") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  theme(plot.title = element_text(size = 11))+
  theme_bw()+
  theme(plot.title = element_blank())

ggsave(filename = "figures/shortest_complexity_grid_boxplots.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

show(plot)


subset_data <- subset(data, gridlike_group == "0.8-1")
anova_result <- aov(shortest_path_complexity_normalized ~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)

subset_data <- subset(data, gridlike_group == "0.6-0.8")
anova_result <- aov(shortest_path_complexity_normalized ~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)

subset_data <- subset(data, gridlike_group == "0.4-0.6")
anova_result <- aov(shortest_path_complexity_normalized ~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)
# Turn count
plot <- ggplot(data, aes(x = circular_crosscorrelation_dist_weighted_full, y = shortest_path_turn_count, fill = gridlike_group)) +
  geom_boxplot() +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationships between the number of turns in shortest path and alignment.",
       x = "Misalignment measured as distance to best cross-correlation.",
       y = "Nr of turns in shortest path.") +
    scale_x_discrete(breaks = x_labels[seq(1, length(x_labels), by = 2)]) +
    guides(fill = "none")

ggsave(filename = "figures/shortest_turns_grid_boxplots.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

plot(show)

# Scatterplot, shortest path length differences, cosine similarity
plot <- ggplot(data, aes(x = cosine_similarity_weighted, y = shortest_diff_ratio, color = gridlike_group)) +
  geom_point(alpha = 0.7) +  # Use points for scatterplot, alpha for transparency
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationships between shortest path distance ratio by Street Network Orientation",
       x = "Misalignment measured as weighted cosine similarity",
       y = "Ratio between straight line and shortest path",
       color = "Gridlike Group") +  # Add a legend title
  theme_minimal() # Optional: Use a cleaner theme

ggsave(filename = "figures/shortest_cosine_grid_scatterplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

show(plot)

# scatterplot, shortest path complexity, cosine similarity
plot <- ggplot(data, aes(x = cosine_similarity_weighted, y = shortest_path_complexity, color = gridlike_group)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Scatterplot of Shortest path distance ratio by Street Network Orientation.",
       x = "Misalignment measured as weighted cosine similarity.",
       y = "Complexity of shortest path.",
       color = "Gridlike Group") +
  theme_minimal()

ggsave(filename = "figures/shortest_cosine_grid_complexity_scatterplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

show(plot)

summary(data$streets_per_node_avg)
summary(data$intersection_density_km)

data <- data %>%
  mutate(intersection_density_group = cut(intersection_density_km,
                                          breaks = 3,  # Number of bins
                                          include.lowest = TRUE))
data <- data %>%
  mutate(streets_per_node_avg_group = cut(streets_per_node_avg,
                                          breaks = c(2, 2.5, 3, 3.5, 4),
                                          labels = c("2-2.5", "2.5-3", "3-3.5", "3.5-4"),
                                          include.lowest = TRUE))

summary(data$streets_per_node_avg_group[data$gridlike_group=="0.8-1"])



library(tidyr)
# --- pivoted --- 

data_summary <- data %>%
  group_by(gridlike_group, alignment) %>%
  summarise(
    mean_intersection_density = mean(intersection_density_km, na.rm = TRUE),
    mean_complexity = mean(shortest_path_complexity_normalized, na.rm = TRUE)
  ) %>%
  pivot_wider(
    names_from = alignment,
    values_from = c(mean_intersection_density, mean_complexity),
    names_sep = "_"
  )


#--- environmental 3d ----
breaks <- quantile(data$intersection_density_km, probs = seq(0, 1, length.out = 4), na.rm = TRUE)

# Create custom labels showing the range of values
labels <- paste0(round(breaks[-length(breaks)], 2), " - ", round(breaks[-1], 2))

# Apply the custom labels
data <- data %>%
  mutate(intersection_density_group = cut(intersection_density_km, breaks = breaks, labels = labels, include.lowest = TRUE))

subset_data <- subset(data, gridlike_median == "above_median")

plot <- ggplot(subset_data, aes(x = alignment, y = shortest_path_complexity_normalized, fill = intersection_density_group)) +
  geom_boxplot() +
  facet_wrap(~ intersection_density_group, nrow = 1, ncol = 4) +
  labs( x = "Alignment",
       y = "Complexity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())


show(plot)

ggsave(filename = "figures/shortest_alignment_x_icd_gridabovemedian_complexity_boxplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

subset_data <- subset(data, gridlike_median == "below_median")

plot <- ggplot(subset_data, aes(x = alignment, y = shortest_path_complexity_normalized, fill = intersection_density_group)) +
  geom_boxplot() +
  facet_wrap(~ intersection_density_group, nrow = 1, ncol = 4) +
  labs( x = "Alignment",
        y = "Complexity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())


show(plot)
ggsave(filename = "figures/shortest_alignment_x_icd_gridbelowmedian_complexity_boxplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

# average node degree interaction

breaks <- quantile(data$streets_per_node_avg, probs = seq(0, 1, length.out = 4), na.rm = TRUE)

labels <- paste0(round(breaks[-length(breaks)], 2), " - ", round(breaks[-1], 2))

data <- data %>%
  mutate(avg_node_deg_group = cut(streets_per_node_avg, breaks = breaks, labels = labels, include.lowest = TRUE))

subset_data <- subset(data, gridlike_median == "above_median")

plot <- ggplot(subset_data, aes(x = alignment, y = shortest_path_complexity_normalized, fill = avg_node_deg_group)) +
  geom_boxplot() +
  facet_wrap(~ avg_node_deg_group, nrow = 1, ncol = 4) +
  labs( x = "Alignment",
        y = "Complexity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())


show(plot)

ggsave(filename = "figures/shortest_alignment_x_avg_node_deg_group_gridabovemedian_complexity_boxplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

subset_data <- subset(data, gridlike_median == "below_median")

plot <- ggplot(subset_data, aes(x = alignment, y = shortest_path_complexity_normalized, fill = avg_node_deg_group)) +
  geom_boxplot() +
  facet_wrap(~ avg_node_deg_group, nrow = 1, ncol = 4) +
  labs( x = "Alignment",
        y = "Complexity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())


show(plot)
ggsave(filename = "figures/shortest_alignment_x_avg_node_deg_group_gridbelowmedian_complexity_boxplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

library(plotly)


library(scatterplot3d)
x <- data$alignment[data$gridlike_median=="above_median"]
y <- data$intersection_density_km[data$gridlike_median=="above_median"]
z <- data$shortest_path_complexity_normalized[data$gridlike_median=="above_median"]

p <- plot_ly(data, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers",
             marker = list(size = 5, color = ~x, colorscale = "Viridis", symbol = "circle")) %>%
  layout(title = "3D Scatter Plot",
         scene = list(xaxis = list(title = "alignment"),
                      yaxis = list(title = "intersection density"),
                      zaxis = list(title = "complexity")))


show(p)


x <- data$alignment
y <- data$intersection_density_km
z <- data$shortest_path_complexity_normalized

p <- plot_ly(data, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers",
             marker = list(size = 5, color = ~z, colorscale = "Viridis", symbol = "circle")) %>%
  layout(title = "3D Scatter Plot",
         scene = list(xaxis = list(title = "alignment"),
                      yaxis = list(title = "intersection density"),
                      zaxis = list(title = "complexity")))



show(p)

scatterplot3d(x, y, z, main="3D Scatterplot", xlab="X-axis", ylab="Y-axis", zlab="Z-axis",color = "blue")
show(plot)
# --------------- Environmental -----------
plot <- ggplot(data, aes(x = alignment, y = shortest_path_complexity_normalized, fill = gridlike_group)) +
  geom_boxplot() +
  facet_wrap(~ streets_per_node_avg_group, nrow = 2, ncol = 3) +
  labs(title = "Relationship between alignment and the circuity of the shortest route in increasingly grid-like cities.",
       x = "Alignment",
       y = "Circuity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())
show(plot)


plot <- ggplot(data, aes(x = continuous_alignment, y = shortest_path_complexity_normalized, color = intersection_density_group)) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE) +  # Quadratic polynomial
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Non-linear Relationship of Complexity and Alignment",
       x = "Alignment",
       y = "Complexity",
       color = "Streets/Node (Avg)") +
  theme_minimal() +
  scale_color_viridis_d()+
  scale_x_continuous(breaks = unique(data$alignment_numeric))
show(plot)


plot <- ggplot(data, aes(x = streets_per_node_avg, y = shortest_path_complexity_normalized, color = alignment)) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE) +  # Quadratic polynomial
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(
       x = "Avg Node Degree",
       y = "Complexity",
       color = "alignment") +
  theme_minimal() +
  scale_color_viridis_d()+
  theme_bw()+
  theme(plot.title = element_blank())

show(plot)

ggsave(filename = "figures/shortest_alignment_x_avgnodedeg_grid_complexity_lineplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

plot <- ggplot(data, aes(x = intersection_density_km, y = shortest_path_complexity_normalized, color = alignment)) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE) +  # Quadratic polynomial
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(
       x = "Interconnection Density (ICD)",
       y = "Complexity",
       color = "alignment") +
  theme_bw()+
  theme(plot.title = element_blank())

show(plot)
ggsave(filename = "figures/shortest_alignment_x_icd_grid_complexity_lineplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

plot <- ggplot(data, aes(x = streets_per_node_avg, y = shortest_path_complexity_normalized, color = environment_orientation_order)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, size=0.8, linetype = "dashed",color="red") +
  facet_wrap(~ alignment, nrow = 3, ncol = 3) +
  labs(
    x = "Avg Node Degree",
    y = "Complexity",
    color = "grid-likeness") +
  theme_minimal() +
  theme_bw()+
  theme(plot.title = element_blank())

show(plot)

ggsave(filename = "figures/shortest_alignment_x_avgnodedeg_grid_complexity_scatter.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

plot <- ggplot(data, aes(x =intersection_density_km , y = shortest_path_complexity_normalized , color = environment_orientation_order)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, size=0.8, linetype = "dashed",color="red") +
  facet_wrap(~ alignment, nrow = 3, ncol = 3) +
  labs(
    x = "Intersection density",
    y = "Complexity",
    color = "grid-likeness") +
  theme_bw()+
  theme(plot.title = element_blank())

show(plot)

ggsave(filename = "figures/shortest_alignment_x_icd_grid_complexity_scatter.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

print(summary(data$intersection_density_group[data$gridlike_group=="0.4-0.6"]))
