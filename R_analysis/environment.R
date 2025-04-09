data <- readRDS("od_pair_data_clean.rds")

library(ggplot2)
library(dplyr)

# Intersection density correlation test
cor_result <- cor.test(data$intersection_density_km, data$shortest_path_complexity_normalized, method="pearson")
print(cor_result)

cor_result <- cor.test(data$intersection_density_km, data$simplest_path_complexity_normalized, method="pearson")
print(cor_result)

# Average node degree correlation test
cor_result <- cor.test(data$streets_per_node_avg, data$shortest_path_complexity_normalized, method="pearson")
print(cor_result)

cor_result <- cor.test(data$streets_per_node_avg, data$simplest_path_complexity_normalized, method="pearson")
print(cor_result)


subset_data <- subset(data, gridlike_median == "above_median")
cor_result <- cor.test(subset_data$intersection_density_km, data$shortest_path_complexity_normalized, method="pearson")
print(cor_result)

cor_result <- cor.test(subset_data$intersection_density_km, data$simplest_path_complexity_normalized, method="pearson")
print(cor_result)



plot1 <- ggplot(data, aes(x = intersection_density_km, y = shortest_path_complexity_normalized, color = environment_orientation_order)) +
  geom_point(alpha = 0.7) +
  labs(
       x = "Intersection Density (km2)",
       y = "Shortest Path Complexity") +
  geom_smooth(color="red", method = "lm", se = TRUE, size = 0.8, linetype = "dashed") +
  theme() +
  theme_bw() +
  theme(plot.title = element_blank())+
  guides(color = "none")

plot2 <- ggplot(data, aes(x = streets_per_node_avg, y = shortest_path_complexity_normalized, color = environment_orientation_order)) +
  geom_point(alpha = 0.7) +
  labs(
    x = "Average Node Degree",
    y = "Simplest Path Complexity",
    color = "Grid-like value") +
  geom_smooth(color="red", method = "lm", se = TRUE, size = 0.8, linetype = "dashed") +
  theme() +
  theme_bw() +
  theme(plot.title = element_blank()) 
  
library(patchwork)

combined_plot <- plot1 + plot2

show(combined_plot)

ggsave(filename = "figures/shortest_icd_and_avgdeg_scatterplot.pdf", 
       plot = combined_plot, 
       device = "pdf", 
       width = 16, 
       height = 6, 
       units = "in", 
       dpi = 300)



plot <- ggplot(data, aes(x = intersection_density_km, y = simplest_path_complexity_normalized, color = alignment)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationships between least complex path distance / straight line distance ratio.",
       x = "Origin & Destination Pairs Distance from Best Cross-Correlation Match",
       y = "Simplest Path Complexity",
       color = "Gridlike Group") +
  geom_smooth(aes(color = gridlike_group), method = "lm", se = FALSE, size = 0.8, linetype = "dashed") +
  theme() +
  theme_bw() 


show(plot)

ggsave(filename = "figures/simplest_ICD_grid_scatterplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)


# average node degree

plot <- ggplot(data, aes(x = streets_per_node_avg, y = shortest_path_complexity_normalized, color = alignment)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationships between least complex path distance / straight line distance ratio.",
       x = "Average Node Degree",
       y = "Simplest Path Complexity",
       color = "Gridlike Group") +
  geom_smooth(aes(color = gridlike_group), method = "lm", se = FALSE, size = 0.8, linetype = "dashed") +
  theme() +
  theme_bw() +
  theme(plot.title = element_blank()) 


show(plot)

ggsave(filename = "figures/shortest_avgdeg_grid_scatterplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

plot <- ggplot(data, aes(x = streets_per_node_avg, y = simplest_path_complexity_normalized, color = alignment)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Relationships between least complex path distance / straight line distance ratio.",
       x = "Average Node Degree",
       y = "Simplest Path Complexity",
       color = "Gridlike Group") +
  geom_smooth(aes(color = gridlike_group), method = "lm", se = FALSE, size = 0.8, linetype = "dashed") +
  theme() +
  theme_bw() 


show(plot)

ggsave(filename = "figures/simplest_avgdeg_grid_scatterplot.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)

# Interaction between alignment and ICD
library(lme4)
library(effects)
library(lmerTest)
library(flexplot)
data$continuous_alignment <- as.numeric(data$alignment)
print(summary(data$gridlike_group))
print(summary(data$continuous_alignment))
print(summary(data$alignment))
print(summary(data$interconnection_density_km))
data$alignment_sq <- data$continuous_alignment^2

quadratic_model <- lm(shortest_path_complexity_normalized ~ intersection_density_km + I(independent_variable^2), data = data)








mixed_model_int <- lmer(shortest_path_complexity_normalized ~ streets_per_node_avg * alignment + (1|gridlike_group) , data = data)
mixed_model_int_complex <- lmer(shortest_path_complexity_normalized ~ streets_per_node_avg  * (alignment | gridlike_group) , data = data)

model.comparison(mixed_model_add,mixed_model_int)

mixed_model_uncorrelated <- lmer(shortest_path_complexity_normalized ~ streets_per_node_avg * alignment + (alignment || gridlike_group), data = data)

visualize(mixed_model_int,plot = "model",sample=9)
summary(mixed_model)
anova(mixed_model)


flexplot(shortest_path_complexity_normalized ~ intersection_density_km |1 + gridlike_group ,
         data = data, method = "lm", se = FALSE, ghost.line = "black")

plot(allEffects(mixed_model))

is.numeric(data$intersection_density_km)

data$predicted_complexity <- predict(mixed_model)


mixed_model_int <- lmer(shortest_path_complexity_normalized ~ streets_per_node_avg * continuous_alignment + (1| gridlike_group), data = data)
mixed_model_add <- lmer(shortest_path_complexity_normalized ~ streets_per_node_avg + continuous_alignment + (1| gridlike_group), data = data)

anova(mixed_model_int)
summary(mixed_model_int)
model.comparison(mixed_model_int,mixed_model_add)

library(ggeffects)


predictions <- ggpredict(mixed_model_int, terms = c("continuous_alignment [all]", "streets_per_node_avg", "gridlike_group [all]"))

ggplot(predictions, aes(x = x, y = predicted, color = group, ymin = conf.low, ymax = conf.high)) +
  geom_line() +
  geom_ribbon(aes(fill = group), alpha = 0.2, colour = NA) +
  facet_wrap(~ facet) +
  labs(x = "Alignment", y = "Predicted Complexity", color = "Streets/Node (Avg)", fill = "Streets/Node (Avg)") +
  theme_bw() 








predictions <- ggpredict(mixed_model, terms = c("continuous_alignment [all]", "streets_per_node_avg", "gridlike_group"))
summary(mixed_model)
ggplot(predictions, aes(x = x, y = predicted, color = group, ymin=conf.low, ymax=conf.high)) +  # Use 'group' for different streets_per_node_avg
  geom_line() +
  geom_ribbon(aes(fill = group), alpha = 0.2, colour = NA) + # Add confidence intervals, remove the line around them
  facet_wrap(~ facet) +  # Facet by 'facet' (which corresponds to gridlike_groups)
  labs(x = "continuous_alignment", y = "Predicted Complexity", color = "streets_per_node_avg", fill = "streets_per_node_avg") +  # Add labels
  theme_bw() 

ggplot(data, aes(x = continuous_alignment, y = shortest_path_complexity_normalized, color = intersection_density_km)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", aes(fill=intersection_density_km), se=TRUE) +
  facet_grid(gridlike_group ~ continuous_alignment, scales = "free") +  # Facet by BOTH group and original alignment
  scale_color_viridis_c() +
  scale_fill_viridis_c() +
  labs(title = "Interaction Plot (Continuous Alignment, by Original Category)",
       x = "Continuous Alignment",
       y = "Shortest Path Complexity (Normalized)",
       color = "Intersection Density",
       fill = "Intersection Density") +
  theme_bw()

env_data <- data %>%
  mutate(streets_per_node_avg_group = cut(streets_per_node_avg,
                                          breaks = c(2, 2.5, 3, 3.5, 4),
                                          labels = c("2-2.5", "2.5-3", "3-3.5", "3.5-4"),
                                          include.lowest = TRUE))

ggplot(env_data, aes(x = alignment, y = shortest_path_complexity_normalized, color = streets_per_node_avg_group)) +
  geom_point(alpha = 0.5) +  # Use semi-transparent points to see density
  geom_smooth(method = "loess", se = TRUE) + # Add smoothed lines (LOESS)
  facet_wrap(~ gridlike_group) + # Facet by gridlike_group
  labs(x = "Alignment", y = "Complexity", color = "Streets/Node (Avg)") +
  theme_bw() +
  scale_color_viridis_d() #Discrete viridis color scale


