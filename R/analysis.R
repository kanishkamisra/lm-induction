library(tidyverse)
library(broom)
library(ggtext)
library(lme4)
library(lmerTest)

theme_misra <-
  theme_bw(base_size = 17, base_family = "Times New Roman") +
  theme(
    legend.title = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black"),
    legend.text = element_markdown(),
    # panel.grid = element_blank()
  )


theme_misra_no_grid <-
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    legend.title = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black"),
    panel.grid = element_blank(),
    legend.text = element_markdown(),
  )

final_results <- bind_rows(
  fs::dir_ls("data/results/", regexp = "-induction-tg.csv") %>%
    map_df(read_csv)
) %>%
  # group_by(model, property) %>%
  # # mutate(n = rep(1:5, 60)) %>%
  # ungroup() %>%
  pivot_longer(
    contains(c("overlap", "similarity", "logprob")),
    names_to = c("metric", "generalization"),
    names_pattern = "(.*)_(.*)"
    # c(overlap_within, overlap_similar, overlap_random, logprob_within, log_prob)
  ) %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  mutate(
    model = case_when(
      model == "axxl-property" ~ "ALBERT-xxl",
      model == "rl-property" ~ "RoBERTa-large",
      model == "bl-property" ~ "BERT-large"
    ),
    category = str_remove(category, "\\.n\\.01")
  ) %>%
  # filter(category != "mollusk") %>%
  mutate(generalization = factor(generalization, levels = c("within", "similar", "random"), labels = c("Within", "Similar", "Random")))


final_results %>%
  ggplot(aes(similarity, exp(logprob), color = generalization, fill = generalization, group = generalization)) +
  geom_point(alpha = 0.05) +
  geom_smooth(method = "lm") +
  facet_grid(model ~ property) +
  scale_color_brewer(type = "qual", palette = "Dark2", direction = -1, aesthetics = c("color", "fill"), labels = c("Within", "Outside<sub><i>similar</i></sub>", "Outside<sub><i>random</i></sub>")) +
  theme_misra

final_results %>%
  group_by(model) %>%
  # mutate(
  #   logprob = (logprob - min(logprob))/(max(logprob) - min(logprob))
  # ) %>%
  ungroup() %>%
  # group_by(model,  n, generalization, property) %>%
  group_by(model,  n, generalization) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    log_prob = mean(logprob)
  ) %>%
  ungroup() %>%
  ggplot(aes(n, log_prob, color = generalization, group = generalization, fill = generalization)) +
  geom_point(size = 2) +
  geom_line(size = 0.7) +
  geom_ribbon(aes(ymin = log_prob - ste, ymax = log_prob + ste), color = NA, alpha = 0.2) +
  # facet_grid(model ~ property) +
  facet_wrap(~model) +
  # scale_y_continuous(breaks = c(0.7, 0.75, 0.8, 0.85, 0.9, 0.95), limits = c(0.70, 0.96)) +
  scale_color_brewer(type = "qual", palette = "Dark2", direction = -1, aesthetics = c("color", "fill"), labels = c("Within", "Outside<sub><i>similar</i></sub>", "Outside<sub><i>random</i></sub>")) +
  # scale_color_manual(values = c("#5e3c99", "#e66101", "#417D7A"), aesthetics = c("color", "fill")) +
  theme_misra +
  labs(
    x = "Number of Adaptation Concepts",
    y = "Generalization"
  )

ggsave("figures/generalization.pdf", width = 5.2, height = 3.7, device = cairo_pdf, dpi = 300)

final_results %>%
  group_by(model, n) %>%
  rstatix::games_howell_test(logprob ~ generalization)

final_results %>%
  group_by(model) %>%
  # mutate(overlap = (overlap - mean(overlap))/sd(overlap), similarity = (similarity - mean(similarity))/sd(similarity)) %>%
  mutate(overlap = scale(overlap), similarity = scale(similarity)) %>%
  nest() %>%
  mutate(
    fit = map(data, function(x) {
      lm(logprob ~ n + overlap * similarity, data = x)
    })
  ) %>%
  pull(fit) %>%
  map_df(glance)

# final_results %>%
#   filter(property == "can dax") %>%
#   group_by(model, category, n, generalization) %>%
#   summarize(
#     ste = 1.96 * plotrix::std.error(overlap),
#     overlap = mean(overlap)
#   ) %>%
#   ggplot(aes(n, overlap, color = generalization, group = generalization)) +
#   geom_point() +
#   geom_line() +
#   facet_grid(category~model)

fit <- lm(logprob ~ n + overlap * similarity, data = x)
summary(fit)

aov_fit <- aov(logprob ~ generalization, data = x %>% filter(n == 5))
summary(aov_fit)
TukeyHSD(aov_fit)
pairwise.t.test(x %>% filter(n == 5) %>% pull(logprob), x %>% filter(n == 5) %>% pull(generalization))

oneway.test(logprob ~ generalization, data = x %>% filter(n == 1), var.equal = FALSE)

final_results %>%
  filter(property == "can dax") %>%
  group_by(model, generalization) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(similarity),
    similarity = mean(similarity)
  ) %>%
  ggplot(aes(model, similarity, color = generalization, fill = generalization)) +
  geom_col(position = position_dodge(), alpha = 0.8) +
  geom_linerange(aes(ymin = similarity - ste, ymax = similarity + ste), position = position_dodge(width = 0.9), color = "black") +
  scale_color_brewer(type = "qual", palette = "Dark2", direction = -1, aesthetics = c("color", "fill"), labels = c("Within", "Outside<sub><i>similar</i></sub>", "Outside<sub><i>random</i></sub>")) +
  scale_y_continuous(limits = c(0, 0.8)) +
  theme_misra_no_grid +
  labs(
    x = "Model",
    y = "Cosine Similarity with\nAdaptation Concepts"
  )

ggsave("figures/cosine.pdf", height = 4.5, width = 5, device = cairo_pdf)

final_results %>%
  filter(property == "can dax") %>%
  group_by(model, n, generalization) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(overlap),
    overlap = mean(overlap)
  ) %>%
  ggplot(aes(model, overlap, color = generalization, fill = generalization)) +
  geom_col(position = position_dodge(), alpha = 0.8) +
  geom_linerange(aes(ymin = overlap - ste, ymax = overlap + ste), position = position_dodge(width = 0.9), color = "black") +
  scale_color_brewer(type = "qual", palette = "Dark2", direction = -1, aesthetics = c("color", "fill"), labels = c("Within", "Outside<sub><i>similar</i></sub>", "Outside<sub><i>random</i></sub>")) +
  # scale_y_continuous(limits = c(0, 0.8)) +
  facet_wrap(~ n) +
  theme_misra_no_grid +
  labs(
    x = "Model",
    y = "Property Overlap with\nAdaptation Concepts"
  )


x <- final_results %>%
  filter(model == "ALBERT-xxl") %>%
  group_by(property) %>%
  mutate(overlap = scale(overlap, scale = FALSE), similarity = scale(similarity, scale = FALSE)) %>%
  ungroup()

fit <- lm(logprob ~ n + overlap * similarity, data = x)
summary(fit)


null <- lmer(logprob ~ n + overlap + similarity + (1|property), data = x)
summary(null)
fit <- lmer(logprob ~ n + overlap * similarity + (1|property) , data = x)
summary(fit)

anova(null, fit)

# teasing apart overlaps and category membership

tease <- read_csv("data/results/tease_apart.csv") %>%
  mutate(
    model = case_when(
      model == "axxl-property" ~ "ALBERT-xxl",
      model == "rl-property" ~ "RoBERTa-large",
      model == "bl-property" ~ "BERT-large"
    )
  )

tease %>%
  pivot_longer(within:outside, names_to = "conclusion", values_to = "logprob") %>%
  mutate(
    conclusion = factor(conclusion, levels = c("within", "outside"), labels = c("Within", "Outside"))
  ) %>%
  group_by(model, conclusion) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    logprob = mean(logprob)
  ) %>%
  ggplot(aes(model, logprob, color = conclusion, fill = conclusion, group = conclusion)) +
  geom_col(position = position_dodge(), alpha = 0.8, width=0.7) +
  geom_linerange(aes(ymin = logprob - ste, ymax = logprob + ste), position = position_dodge(width = 0.7), color = "black") +
  scale_color_manual(values = c("#7570b3", "#d95f02"), aesthetics = c("color", "fill")) +
  # scale_color_brewer(type = "qual", palette = "Dark2", direction = -1, aesthetics = c("color", "fill")) +
  scale_y_continuous(limits = c(-0.41, 0)) +
  theme_misra_no_grid +
  labs(
    x = "Model",
    y = "Generalization"
  )

ggsave("figures/teaseapart.pdf", width = 5, height = 3, device = cairo_pdf, dpi = 300)


test_results <- tease %>%
  group_by(model) %>%
  nest() %>%
  mutate(
    t_test = map(data, function(x) {
      t.test(x$within, x$outside, paired = TRUE) %>%
        tidy()
    })
  ) %>%
  select(-data) %>%
  unnest(t_test)

# benjamini-hochberg FDR correction.
p.adjust(test_results$p.value, method = "BH")
