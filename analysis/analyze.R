require(corrplot)
require(ggplot2)
require(ggpubr)
require(effsize)
require(xtable)
require(ScottKnott)
require(gtools)
require(stringi)
require(stringr)
require(scales)
require(tidyr)
require(dplyr)
require(forcats)
#devtools::install_github("klainfo/ScottKnottESD")


# setting working directory to the project directory
setwd('~/suppmaterial-25-yingzhe-AIOpsSelection/')

# variables
datasets <- c('google', 'backblaze', 'alibaba')
periods <- c(28, 36, 8)
names(periods) <- datasets
models <- c('rf', 'nn', 'cart', 'lr')


for (dataset in datasets) {
  # data loading and manipulation
  file_list <- paste('./results/selection_', dataset, '_', models, '.csv', sep='')
  df_list <- lapply(file_list, read.csv)
  df <- do.call('rbind', df_list)
  levels(df$Scenario) <- list(
    Stationary = 'stationary',
    Retrain = 'retrain',
    Oracle = 'oracle',
    LaF = 'laf',
    CRC = 'crc',
    TBM = 'temporal',
    rTBM = 'temporal_rev',
    SBM = 'dist',
    rSBM = 'dist_leak'
  )


  # AUC performance trend line plot
  ggplot(df %>% 
    select(Scenario, Model, Testing.Period, Test.AUC) %>%
    group_by(Scenario, Model, Testing.Period) %>% 
    summarize(AUC=mean(Test.AUC)),
    #aes(x=factor(Testing.Period), y=AUC, group=Scenario, color=Scenario, linetype=Scenario)) +
    aes(x=Testing.Period, y=AUC, group=Scenario, color=Scenario, shape=Scenario)) + 
    geom_line(size=0.2) + geom_point(size=0.8) + ylim(0.5, 1) + 
    theme(legend.key.size=unit(4, 'mm'), legend.text=element_text(size=8), legend.title=element_text(size=8)) + 
    facet_grid(.~toupper(Model)) + 
    labs(x='Testing Time Period', y='AUC', shape='Mechanism', color='Mechanism') +
    #scale_x_discrete(labels=as.integer(as.integer(periods[dataset])/2+1):as.integer(periods[dataset])) +
    scale_shape_manual(values=c(16, 16, 16, 17, 17, 17, 18, 17, 18))
  
  ggsave(paste('selection_model_auc_trend_', dataset, '.pdf', sep=''), width=200, height=50, units='mm', dpi=300)


  # Scott-Knott clustering boxplot of AUC performance 
  df_sk <- df %>% 
    group_by(Scenario, Model, Round) %>% 
    summarise(AUC = mean(Test.AUC)) %>%
    mutate(Label = paste(Scenario, toupper(Model), sep='/'))
  sk <- with(df_sk, SK(x=Label, y=AUC, model='y~x', which='x'))
  sk <- summary(sk)
  df_sk <- merge(df_sk, data.frame(Label=sk$Levels, Group=as.integer(sk$`SK(5%)`)), by='Label')

  ggplot(df_sk, aes(x=str_wrap(Label, 20), y=AUC, color=toupper(Model))) + 
    geom_boxplot(position=position_dodge(width=0.1)) + 
    facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 50, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='Selection mechanism', y='AUC', color='Model') + ylim(0.5, 1)

  ggsave(paste('selection_model_auc_sk_', dataset, '.pdf', sep=''), width=200, height=50, units='mm', dpi=300)
}



df <- read.csv('./results/ranking_analysis.csv')
levels(df$Scenario) <- list(
  Stationary = 'stationary',
  Retrain = 'retrain',
  Oracle = 'oracle',
  LaF = 'laf',
  CRC = 'crc',
  TBM = 'temporal',
  rTBM = 'temporal_rev',
  SBM = 'dist',
  rSBM = 'dist_leak'
)


for (dataset in datasets) {
  ggplot(df %>% 
    filter(Dataset == dataset & Round != -1 & Testing.Period > periods[dataset]/2+1) %>%
    select(Scenario, Model, Testing.Period, Kendall.Tau) %>%
    group_by(Scenario, Model, Testing.Period) %>% 
    summarize(Measure=mean(abs(Kendall.Tau))),
    #aes(x=Testing.Period, y=Measure, group=Scenario, color=Scenario, shape=Scenario)) +
    aes(x=factor(Testing.Period), y=Measure, group=Scenario, color=Scenario, shape=Scenario)) +
    geom_line(size=0.2) + geom_point(size=0.8) + ylim(0, 1) + 
    facet_grid(.~toupper(Model)) + 
    labs(x='Testing Time Period', y='Kendall\'s Tau', shape='Mechanism', color='Mechanism') +
    #scale_x_discrete(labels=as.integer(as.integer(periods[dataset])/2+1):as.integer(periods[dataset])) +
    scale_shape_manual(values=c(17, 17, 17, 18, 17, 18))

  ggsave(paste('selection_model_tau_trend_', dataset, '.pdf', sep=''), width=200, height=50, units='mm', dpi=300)

  df_sk <- df %>% 
    filter(Dataset == dataset & Round != -1) %>%
    group_by(Scenario, Model, Round) %>% 
    summarise(Measure = mean(abs(Kendall.Tau))) %>%
    mutate(Label = paste(Scenario, toupper(Model), sep='/'))
  sk <- with(df_sk, SK(x=Label, y=Measure, model='y~x', which='x'))
  sk <- summary(sk)
  df_sk <- merge(df_sk, data.frame(Label=sk$Levels, Group=as.integer(sk$`SK(5%)`)), by='Label')

  ggplot(df_sk, aes(x=str_wrap(Label, 20), y=Measure, color=toupper(Model))) + 
    geom_boxplot(position=position_dodge(width=0.1)) + 
    geom_hline(yintercept = 0.6, color = 'red') +
    geom_hline(yintercept = 0.3, color = 'red') +
    facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 50, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='Selection mechanism', y='Kendall\'s Tau', color='Model') + ylim(0, 1)

  ggsave(paste('selection_model_tau_sk_', dataset, '.pdf', sep=''), width=200, height=50, units='mm', dpi=300)

  ggplot(df %>% 
    filter(Dataset == dataset & Round != -1 & Testing.Period > periods[dataset]/2+1) %>%
    select(Scenario, Model, Testing.Period, Jaccard.3) %>%
    group_by(Scenario, Model, Testing.Period) %>% 
    summarize(Measure=mean(Jaccard.3)),
    #aes(x=Testing.Period, y=AUC, group=Scenario, color=Scenario, linetype=Scenario)) +
    aes(x=factor(Testing.Period), y=AUC, group=Scenario, color=Scenario, linetype=Scenario)) +
    aes(x=Testing.Period, y=Measure, group=Scenario, color=Scenario, shape=Scenario) +
    geom_line(size=0.2) + geom_point(size=0.8) + ylim(0, 1) + 
    facet_grid(.~toupper(Model)) + 
    labs(x='Testing Time Period', y='Jaccard', shape='Mechanism', color='Mechanism') +
    #scale_x_discrete(labels=as.integer(as.integer(periods[dataset])/2+1):as.integer(periods[dataset])) +
    scale_shape_manual(values=c(17, 17, 17, 18, 17, 18))

  ggsave(paste('selection_model_jaccard_trend_', dataset, '.pdf', sep=''), width=200, height=50, units='mm', dpi=300)

  df_sk <- df %>% 
    filter(Dataset == dataset & Round != -1) %>%
    group_by(Scenario, Model, Round) %>% 
    summarise(Measure = mean(abs(Jaccard.3))) %>%
    mutate(Label = paste(Scenario, toupper(Model), sep='/'))
  sk <- with(df_sk, SK(x=Label, y=Measure, model='y~x', which='x'))
  sk <- summary(sk)
  df_sk <- merge(df_sk, data.frame(Label=sk$Levels, Group=as.integer(sk$`SK(5%)`)), by='Label')

  ggplot(df_sk, aes(x=str_wrap(Label, 20), y=Measure, color=toupper(Model))) + 
    geom_boxplot(position=position_dodge(width=0.1)) + 
    facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 50, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    labs(x='Selection mechanism', y='Jaccard', color='Model') + ylim(0, 1)

  ggsave(paste('selection_model_jaccard_sk_', dataset, '.pdf', sep=''), width=200, height=50, units='mm', dpi=300)

  ggplot(df %>% 
    filter(Dataset == dataset & Round == -1 & Testing.Period > periods[dataset]/2+1) %>%
    select(Scenario, Model, Testing.Period, Kendall.W) %>%
    group_by(Scenario, Model, Testing.Period),
    aes(x=factor(Testing.Period), y=Kendall.W, group=Scenario, color=Scenario, shape=Scenario)) +
    #aes(x=Testing.Period, y=Kendall.W, group=Scenario, color=Scenario, shape=Scenario)) +
    geom_line(size=0.2) + geom_point(size=0.8) + ylim(0, 1) + 
    facet_grid(.~toupper(Model)) + 
    labs(x='Testing Time Period', y='Kendall\'s W', shape='Mechanism', color='Mechanism') +
    #scale_x_discrete(labels=as.integer(as.integer(periods[dataset])/2+1):as.integer(periods[dataset])) +
    scale_shape_manual(values=c(17, 17, 17, 18, 17, 18))
  ggsave(paste('selection_model_w_trend_', dataset, '.pdf', sep=''), width=200, height=50, units='mm', dpi=300)

  df_sk <- df %>% 
    filter(Dataset == dataset & Round == -1) %>%
    select(Scenario, Model, Kendall.W) %>%
    mutate(Label = paste(Scenario, toupper(Model), sep='/'), Measure=Kendall.W)
  sk <- with(df_sk, SK(x=Label, y=Measure, model='y~x', which='x'))
  sk <- summary(sk)
  df_sk <- merge(df_sk, data.frame(Label=sk$Levels, Group=as.integer(sk$`SK(5%)`)), by='Label')

  ggplot(df_sk, aes(x=str_wrap(Label, 20), y=Measure, color=toupper(Model))) + 
    geom_boxplot(position=position_dodge(width=0.1)) + 
    facet_grid(.~Group, scales='free_x', space = "free_x") + 
    theme(axis.text.x = element_text(angle = 50, hjust = 1)) +
    theme(axis.text=element_text(size=6)) +
    geom_hline(yintercept = 0.6, color = 'red') +
    geom_hline(yintercept = 0.3, color = 'red') +
    labs(x='Selection mechanism', y='Kendall\'s W', color='Model') + ylim(0, 1)
  ggsave(paste('selection_model_w_sk_', dataset, '.pdf', sep=''), width=200, height=50, units='mm', dpi=300)
}
