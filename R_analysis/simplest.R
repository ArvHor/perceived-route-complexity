
data <- readRDS("od_pair_data_clean.rds")
library(ggplot2)
library(dplyr)

## ----- Simplest path ----------------

# Simplest path length differences
plot <- ggplot(data, aes(x = alignment, y = simplest_diff_ratio, fill = gridlike_group)) +
  geom_boxplot() +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationships between alignment and the circuity of the simplest route in increasingly grid-like cities.",
       x = "Alignment",
       y = "Circuity") +
  guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())

ggsave(filename = "figures/simplest_circuity_grid_boxplots.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 600)

plot(show)
subset_data <- subset(data, gridlike_group == "0.8-1")
anova_result <- aov(simplest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)

subset_data <- subset(data, gridlike_group == "0.6-0.8")
anova_result <- aov(simplest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)
subset_data <- subset(data, gridlike_group == "0.4-0.6")
anova_result <- aov(simplest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)
subset_data <- subset(data, gridlike_group == "0.2-0.4")
anova_result <- aov(simplest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)
subset_data <- subset(data, gridlike_group == "0-0.2")
anova_result <- aov(simplest_diff_ratio~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)
#Complexity and alignment across environments
plot <- ggplot(data, aes(x = alignment, y = simplest_path_complexity_normalized, fill = gridlike_group)) +
  geom_boxplot() +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(
       x = "Alignment.",
       y = "Complexity") +
  guides(fill = "none") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  theme(plot.title = element_text(size = 11)) +
  theme_bw() +
  theme(plot.title = element_blank())

ggsave(filename = "figures/simplest_complexity_grid_boxplots.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)
show(plot)


subset_data <- subset(data, gridlike_group == "0.8-1")
anova_result <- aov(simplest_path_complexity_normalized ~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)

subset_data <- subset(data, gridlike_group == "0.6-0.8")
anova_result <- aov(simplest_path_complexity_normalized ~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)

subset_data <- subset(data, gridlike_group == "0.4-0.6")
anova_result <- aov(simplest_path_complexity_normalized ~ alignment, data = subset_data)
summary(anova_result)
TukeyHSD(anova_result)

plot <- ggplot(data, aes(x = cosine_similarity_weighted, y = simple_straight_diff, color = gridlike_group)) +
  geom_point(alpha = 0.7) +  
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationships between least complex path distance / straight line distance ratio and cosine similarity.",
       x = "Origin & Destination Pairs Distance from Best Cross-Correlation Match",
       y = "Difference between Straight Line and Simplest Path",
       color = "Gridlike Group") +
  theme_minimal() 

ggsave(filename = "figures/simplest_cosine_ratio_grid_scatterplot.svg", 
       plot = plot, 
       device = "svg", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)


# Scatterplot for simplest_path_complexity
plot <- ggplot(data, aes(x = cosine_similarity_weighted, y = simplest_path_complexity, color = gridlike_group)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, size=0.8, linetype = "dashed") +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationships between least complex path distance / straight line distance ratio.",
       x = "Origin & Destination Pairs Distance from Best Cross-Correlation Match",
       y = "Simplest Path Complexity",
       color = "Gridlike Group") +
  theme_minimal()

ggsave(filename = "figures/simplest_cosine_complexity_grid_scatterplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)


plot <- ggplot(data, aes(x = continuous_alignment, y = simplest_path_complexity_normalized, color = streets_per_node_avg_group)) +
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


plot <- ggplot(data, aes(x = continuous_alignment, y = simplest_path_complexity_normalized, color = intersection_density_group)) +
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


# icd interaction

breaks <- quantile(data$intersection_density_km, probs = seq(0, 1, length.out = 4), na.rm = TRUE)

# Create custom labels showing the range of values
labels <- paste0(round(breaks[-length(breaks)], 2), " - ", round(breaks[-1], 2))

# Apply the custom labels
data <- data %>%
  mutate(intersection_density_group = cut(intersection_density_km, breaks = breaks, labels = labels, include.lowest = TRUE))

subset_data <- subset(data, gridlike_median == "above_median")

plot <- ggplot(subset_data, aes(x = alignment, y = simplest_path_complexity_normalized, fill = intersection_density_group)) +
  geom_boxplot() +
  facet_wrap(~ intersection_density_group, nrow = 1, ncol = 4) +
  labs( x = "Alignment",
        y = "Complexity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())


show(plot)

ggsave(filename = "figures/simplest_alignment_x_icd_gridabovemedian_complexity_boxplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

subset_data <- subset(data, gridlike_median == "below_median")

plot <- ggplot(subset_data, aes(x = alignment, y = simplest_path_complexity_normalized, fill = intersection_density_group)) +
  geom_boxplot() +
  facet_wrap(~ intersection_density_group, nrow = 1, ncol = 4) +
  labs( x = "Alignment",
        y = "Complexity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())


show(plot)
ggsave(filename = "figures/simplest_alignment_x_icd_gridbelowmedian_complexity_boxplot.pdf", 
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

plot <- ggplot(subset_data, aes(x = alignment, y = simplest_path_complexity_normalized, fill = avg_node_deg_group)) +
  geom_boxplot() +
  facet_wrap(~ avg_node_deg_group, nrow = 1, ncol = 4) +
  labs( x = "Alignment",
        y = "Complexity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())


show(plot)

ggsave(filename = "figures/simplest_alignment_x_avg_node_deg_group_gridabovemedian_complexity_boxplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

subset_data <- subset(data, gridlike_median == "below_median")

plot <- ggplot(subset_data, aes(x = alignment, y = simplest_path_complexity_normalized, fill = avg_node_deg_group)) +
  geom_boxplot() +
  facet_wrap(~ avg_node_deg_group, nrow = 1, ncol = 4) +
  labs( x = "Alignment",
        y = "Complexity")+guides(fill = "none")+
  theme(plot.title = element_text(size = 11))+
  theme_bw() +
  theme(plot.title = element_blank())


show(plot)
ggsave(filename = "figures/simplest_alignment_x_avg_node_deg_group_gridbelowmedian_complexity_boxplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)


## environment interaction

plot <- ggplot(data, aes(x = streets_per_node_avg, y = simplest_path_complexity_normalized, color = alignment)) +
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

ggsave(filename = "figures/simplest_alignment_x_avgnodedeg_grid_complexity_lineplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

plot <- ggplot(data, aes(x = intersection_density_km, y = simplest_path_complexity_normalized, color = alignment)) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE) +  # Quadratic polynomial
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(
    x = "Interconnection Density (ICD)",
    y = "Complexity",
    color = "alignment") +
  theme_bw()+
  theme(plot.title = element_blank())

show(plot)
ggsave(filename = "figures/simplest_alignment_x_icd_grid_complexity_lineplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

# scatterplots

plot <- ggplot(data, aes(x = streets_per_node_avg, y = simplest_path_complexity_normalized, color = environment_orientation_order)) +
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

ggsave(filename = "figures/simplest_alignment_x_avgnodedeg_grid_complexity_scatter.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

plot <- ggplot(data, aes(x =intersection_density_km , y = simplest_path_complexity_normalized , color = environment_orientation_order)) +
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

ggsave(filename = "figures/simplest_alignment_x_icd_grid_complexity_scatter.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

# boxplots again



