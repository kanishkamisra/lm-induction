library(tidyverse)
library(broom)

final_results <- bind_rows(
  fs::dir_ls("data/results/") %>%
    map_df(read_csv)
) %>%
  group_by(model, property) %>%
  mutate(n = rep(1:5, 60)) %>%
  ungroup() %>%
  pivot_longer(
    contains(c("overlap", "logprob")),
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
  filter(category != "mollusk")

final_results %>%
  group_by(model) %>%
  mutate(
    logprob = (logprob - min(logprob))/(max(logprob) - min(logprob))
  ) %>%
  ungroup() %>%
  group_by(model,  n, generalization) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    log_prob = mean(logprob)
  ) %>%
  ungroup() %>%
  mutate(generalization = factor(generalization, levels = c("within", "similar", "random"), labels = c("Within", "Similar", "Random"))) %>%
  ggplot(aes(n, log_prob, color = generalization, group = generalization, fill = generalization)) +
  geom_point(size = 2) +
  geom_line(size = 0.7) +
  geom_ribbon(aes(ymin = log_prob - ste, ymax = log_prob + ste), color = NA, alpha = 0.2) +
  facet_wrap(~model) +
  scale_color_brewer(type = "qual", palette = "Dark2", direction = -1, aesthetics = c("color", "fill")) +
  # scale_color_manual(values = c("#5e3c99", "#e66101", "#417D7A"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    legend.position = "top",
    legend.title = element_blank()
  ) +
  labs(
    x = "n",
    y = "G"
  )

x <- final_results %>%
  filter(model == "ALBERT-xxl") %>%
  mutate(overlap = overlap - mean(overlap))

fit <- lm(logprob ~ n + overlap + factor(generalization), data = x)
summary(fit)


unconfounded <- final_results %>%
  group_by(model) %>%
    mutate(overlap_c = overlap - mean(overlap)) %>%
    nest() %>%
    mutate(
      fits = map(data, function(d) {
        fit <- lm(logprob ~ overlap_c, data = d)
        
        corrected_score = residuals(fit) + coef(fit)["(Intercept)"]
        p_value = glance(fit)$p.value
        tidied <- tidy(fit)
        beta_1 = tidied %>% filter(term == "overlap_c") %>% pull(estimate)
        p_feature = tidied %>% filter(term == "overlap_c") %>% pull(`p.value`)
        
        # beta_1 = tidied %>% filter(term == "cosine_c") %>% pull(estimate)
        # p_feature = tidied %>% filter(term == "cosine_c") %>% pull(`p.value`)
        
        return(
          tibble(u_log_prob = corrected_score, rsq = summary(fit)$r.squared, p_feature = p_feature, p = p_value, beta1 = beta_1)
        )
      })
    ) %>%
    unnest(c(data, fits)) %>%
    ungroup()

unconfounded %>% group_by(model) %>%
  summarize(
    rsq = mean(rsq),
    p = mean(p),
    beta1 = mean(beta1)
  )

unconfounded %>%
  mutate(logprob = exp(u_log_prob)) %>%
  # group_by(model) %>%
  # mutate(
  #   logprob = (logprob - min(logprob))/(max(logprob) - min(logprob))
  # ) %>%
  ungroup() %>%
  group_by(model,  n, generalization) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    log_prob = mean(logprob)
  ) %>%
  ungroup() %>%
  mutate(generalization = factor(generalization, levels = c("within", "similar", "random"), labels = c("Within", "Similar", "Random"))) %>%
  ggplot(aes(n, log_prob, color = generalization, group = generalization, fill = generalization)) +
  geom_point(size = 2) +
  geom_line(size = 0.7) +
  geom_ribbon(aes(ymin = log_prob - ste, ymax = log_prob + ste), color = NA, alpha = 0.2) +
  facet_wrap(~model) +
  scale_color_brewer(type = "qual", palette = "Dark2", direction = -1, aesthetics = c("color", "fill")) +
  # scale_color_manual(values = c("#5e3c99", "#e66101", "#417D7A"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    legend.position = "top",
    legend.title = element_blank()
  ) +
  labs(
    x = "n",
    y = "G"
  )

