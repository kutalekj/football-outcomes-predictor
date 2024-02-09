library(tidyr)
library(dplyr)
library(agricolae)
library(ggplot2)
library(base)

# Read and reshape data
dat_raw <- read.csv2(file = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\matches_10_10.csv", header = TRUE, sep = ",")

dat <- dat_raw |> mutate(
  across(
    c(team_home, team_away, country, competition, season, referee, neutral_field, finished, no_spectators), factor))
dat <- dat |> mutate(
  across(
    where(is.character) & (starts_with("odd") | starts_with("expected")), as.numeric))
dat <- dat |> mutate(
  across(
    where(is.character) & !c(id, date_time), as.integer))
dat$date_time <- as.POSIXct(dat$date_time, format="%Y-%m-%d %H:%M", tz="UTC")
dat <- dat |> mutate(
  competition = gsub(":", "_", competition)
)
dat <- dat |> mutate(
  total_goals = goals_home + goals_away
)

# New column for match group (country, competition, season)
dat <- dat |> mutate(
  match_group = factor(paste0(country, "_", competition, "_", season)))
# dat$match_group <- gsub(":", "_", dat$match_group) # prevent issues with the ":" symbol for "FORTUNA:LIGA"

# New columns for margin
dat <- dat |> mutate(
  margin_tipsport_start = ifelse(
    is.na(odd_tipsport_1_start) | is.na(odd_tipsport_0_start) | is.na(odd_tipsport_2_start) | 
      odd_tipsport_1_start <= 0   | odd_tipsport_0_start <= 0   | odd_tipsport_2_start <= 0,
    NA_real_,
    (1/odd_tipsport_1_start + 1/odd_tipsport_0_start + 1/odd_tipsport_2_start) - 1),
  margin_tipsport_end = ifelse(
    is.na(odd_tipsport_1_end) | is.na(odd_tipsport_0_end) | is.na(odd_tipsport_2_end) | 
      odd_tipsport_1_end <= 0   | odd_tipsport_0_end <= 0   | odd_tipsport_2_end <= 0,
    NA_real_,
    (1/odd_tipsport_1_end + 1/odd_tipsport_0_end + 1/odd_tipsport_2_end) - 1),
  margin_fortuna_start = ifelse(
    is.na(odd_fortuna_1_start) | is.na(odd_fortuna_0_start) | is.na(odd_fortuna_2_start) | 
      odd_fortuna_1_start <= 0   | odd_fortuna_0_start <= 0   | odd_fortuna_2_start <= 0,
    NA_real_,
    (1/odd_fortuna_1_start + 1/odd_fortuna_0_start + 1/odd_fortuna_2_start) - 1),
  margin_fortuna_end = ifelse(
    is.na(odd_fortuna_1_end) | is.na(odd_fortuna_0_end) | is.na(odd_fortuna_2_end) | 
      odd_fortuna_1_end <= 0   | odd_fortuna_0_end <= 0   | odd_fortuna_2_end <= 0,
    NA_real_,
    (1/odd_fortuna_1_end + 1/odd_fortuna_0_end + 1/odd_fortuna_2_end) - 1)
)

# New columns for win/draw/lose probability without margins
dat <- dat |> mutate(
  prb_nomarg_1_tips_end = (1/odd_tipsport_1_end) - (1/odd_tipsport_1_end) * (margin_tipsport_end / (1/odd_tipsport_1_end + 1/odd_tipsport_0_end + 1/odd_tipsport_2_end)),
  prb_nomarg_0_tips_end = (1/odd_tipsport_0_end) - (1/odd_tipsport_0_end) * (margin_tipsport_end / (1/odd_tipsport_1_end + 1/odd_tipsport_0_end + 1/odd_tipsport_2_end)),
  prb_nomarg_2_tips_end = (1/odd_tipsport_2_end) - (1/odd_tipsport_2_end) * (margin_tipsport_end / (1/odd_tipsport_1_end + 1/odd_tipsport_0_end + 1/odd_tipsport_2_end))
  # TODO: For Fortuna (other sports betting company)
)

# New columns for difference in win probabilities (without margin) for both teams
dat <- dat |> mutate(
  diff_prb_nomarg_tips_end = abs(prb_nomarg_1_tips_end - prb_nomarg_2_tips_end)
)

# Filter out matches during the COVID-19 period and those without spectators
no_spectators_matches <- dat |> filter(no_spectators == "True")
filtered <- dat |> filter(no_spectators == "False" & (date_time < "2020-03-01 00:00" | date_time > "2021-09-01 00:00"))

# Selection of possible draw matches
draw_matches <- filtered |> filter(prb_nomarg_0_tips_end >= 0.28 &
                                     prb_nomarg_0_tips_end <= 0.35 &
                                     diff_prb_nomarg_tips_end <= 0.15)

test <- draw_matches |> filter(odd_tipsport_1_end < 2.1 | odd_tipsport_2_end < 2.1)

# Distribute all the selected matches into bins as follows: [2.7-2.8, 2.8-2.9, ..., 3.4-3.5]
draw_matches_cut <- draw_matches |> mutate(
  bin = cut(prb_nomarg_0_tips_end, 
            breaks = c(0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35),
            labels = c('28-29%', '29-30%', '30-31%', '31-32%', '32-33%', '33-34%', '34-35%'),
            include.lowest = TRUE, right = FALSE)
)
table(draw_matches_cut$bin)


# Histograms of total goals for each bin of matches
summaries <- draw_matches_cut |>
  group_by(bin) |>
  summarize(mean_goals = mean(total_goals, na.rm=T),
            variance = var(total_goals, na.rm=T),
            max_count = max(table(total_goals)),  # the highest bar for positioning
            total_matches = n())  # total number of matches in each bin

draw_matches_cut |> ggplot(aes(x=total_goals)) + 
  geom_histogram(binwidth=1, fill="blue", alpha=0.7, color="white") +
  facet_wrap(~bin, scales="free_y") + 
  geom_vline(data=summaries, aes(xintercept=mean_goals),
             color="red", linetype="dashed", size=1) + 
  geom_text(data=summaries, 
            aes(x=mean_goals + 1, y=max_count - 1, 
                label=sprintf("Mean: %.2f\nVar: %.2f\nMatches: %d", mean_goals, variance, total_matches)), 
            hjust=0, vjust=1) +
  labs(title="Histogram of total goals in likely draw resulting matches (all non-COVID matches with spectators)",
       x="Total Goals", y="Number of Matches")

