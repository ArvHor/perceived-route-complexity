
# Load the data
data <- readRDS("od_pair_data_clean.rds")
data_raw <- read.csv("od_pair_data.csv")
library(ggplot2)
library(dplyr)

## ----- Descriptive city-wide statistics ----------------

plot_data <- data %>%
  group_by(city_name) %>%
  summarize(
    avg_env_order = mean(environment_orientation_order),
    count = n()
  )

ggplot(plot_data, aes(x = avg_env_order, y = count)) +
  geom_point(position = position_jitter(width = 0.1, height = 0.2, seed = 123)) + # Jitter the points with a fixed seed
  geom_text(aes(label = city_name), position = position_jitter(width = 0.1, height = 0.2, seed = 123), hjust = -0.2, vjust = 0.5) + # Jitter the labels with the same seed
  labs(title = "City Count vs. Average Environment Orientation Order",
       x = "Average Environment Orientation Order",
       y = "Number of Occurrences")

# Comparison of length differences between shortest and simplest route across environments

data$shortest_simple_length_diff <- data$simplest_diff_ratio - data$shortest_diff_ratio
plot <- ggplot(data, aes(x = alignment, y = shortest_simple_length_diff, fill = gridlike_group)) +
  geom_boxplot() +
  facet_wrap(~ gridlike_group, nrow = 2, ncol = 3) +
  labs(title = "Difference in complexity between shortest and simplest route",
       x = "Misalignment measured as distance to best cross-correlation.",
       y = "Complexity difference") +
  guides(fill = "none") 
show(plot)

ggsave(filename = "figures/shortest_simplest_complexity_diff_grid_boxplots.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)


## ----- Statistical tests ----------------
library(dplyr)

# Are misaligned routes in gridlike networks more complex than routes in non-gridlike networks?

nogrid_vs_highgridmisalign<- data %>%
  filter(grid_alignment_group %in% c("highgrid_misaligned", "nogrid")) %>%
  droplevels()

anova_result <- aov(shortest_path_complexity ~ grid_alignment_group, data = nogrid_vs_highgridmisalign)
summary(anova_result)


nogrid_vs_highgridmisalign<- data %>%
  filter(grid_alignment_group %in% c("highgrid_aligned", "highgrid_misaligned")) %>%
  droplevels()

anova_result <- aov(shortest_path_complexity ~ grid_alignment_group, data = nogrid_vs_highgridmisalign)
summary(anova_result)

anova_result <- aov(simplest_path_complexity ~ grid_alignment_group, data = nogrid_vs_highgridmisalign)
summary(anova_result)

# Are misaligned routes in gridlike networks more complex than aligned routes in gridlike networks?

highgridmisalign_vs_highgridalign <- data %>%
  filter(grid_alignment_group %in% c("highgrid_misaligned", "highgrid_aligned")) %>%
  droplevels()

anova_result <- aov(shortest_path_complexity ~ grid_alignment_group, data = highgridmisalign_vs_highgridalign)
summary(anova_result)
mean_diff_table <- aggregate(shortest_path_complexity ~ grid_alignment_group, 
                             data = highgridmisalign_vs_highgridalign, 
                             FUN = mean)
print(mean_diff_table)

# Calculate the difference directly
mean_diff <- diff(mean_diff_table$shortest_path_complexity)
print(paste("Mean Difference (highgrid_aligned - highgrid_misaligned):", mean_diff))

plot <- ggplot(highgridmisalign_vs_highgridalign, aes(x = grid_alignment_group, y = simplest_path_complexity)) +
  geom_boxplot() +
  labs(title = "Boxplots of Simplest Path Complexity by Alignment Group",
       x = "Misaligned and aligned routes in gridlike networks.",
       y = "Complexity of simplest path.") +
  #scale_x_discrete(breaks = x_labels[seq(1, length(x_labels), by = 2)]) +
  #guides(fill = "none") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.05)) 

ggsave(filename = "figures/simple_grid_alignment_groups_grid_boxplots.pdf", 
       plot = plot, 
       device = "pdf", 
       width = 8, 
       height = 6, 
       units = "in", 
       dpi = 300)
show(plot)


# shortest vs simplest

highgrid_vs_lowgrid <- data %>%
  filter(gridlike_group %in% c("0-0.2", "0.8-1")) %>%
  droplevels()

anova_result <- aov(simplest_path_complexity_normalized ~ gridlike_group, data = highgrid_vs_lowgrid)
summary(anova_result)
TukeyHSD(anova_result)

anova_result <- aov(shortest_path_complexity_normalized ~ gridlike_group, data = highgrid_vs_lowgrid)
summary(anova_result)
TukeyHSD(anova_result)

## Is it uniform?
subset_data_nogrid <- data[data$gridlike_group == "0-0.2", ]
subset_data_highgrid <- data[data$gridlike_group == "0-0.2", ]
split_data <- split(subset_data_nogrid, subset_data_nogrid$circular_crosscorrelation_dist_weighted_9)

data$gridlike_group <- factor(data$gridlike_group, levels = c("0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"))
data$circular_distance_to_4 <- ordered(data$circular_distance_to_4, levels = c(0, 1, 2, 3, 4))
print(summary(data$circular_distance_to_4))
library(MASS)
library(lme4)
library(flexplot)
#model <- lmer(difficulty_rating ~ route_condition + group.Reliance.turn.by.turn + (route_condition|participant), data = data)
#model <- lmer(difficulty_rating ~ route_condition + group.Reliance.turn.by.turn + (route_condition|participant), data = data)
#model_base <- lmer(difficulty_rating ~ route_condition+group.Reliance.turn.by.turn +(1|participant), data = data)
#model_interact <- lmer(difficulty_rating ~ route_condition*group.Reliance.turn.by.turn +(route_condition|participant), data = data)

#model.comparison(model_base, model)
#model.comparison(model_interact, model)
#summary(model)

model <- lm(simplest_path_complexity ~ circular_distance_to_4 + gridlike_group, data = data)
model <- lm(shortest_path_complexity ~ circular_distance_to_4 + gridlike_group, data = data)
visualize(model,plot='model')
summary(model)

model <- lm(simplest_path_complexity ~ circular_distance_to_4, data = subset_data_highgrid)
model <- lm(shortest_path_complexity ~ circular_distance_to_4, data = subset_data_highgrid)
visualize(model,plot='model')
summary(model)



anova_result <- aov(shortest_path_complexity ~ circular_distance_to_4, data = highgrid_vs_lowgrid)
summary(anova_result)
TukeyHSD(anova_result)

