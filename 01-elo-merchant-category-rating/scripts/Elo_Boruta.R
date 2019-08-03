install.packages("Boruta")
library("Boruta")

train_data <- read.csv("train_agg.csv", header = T, stringsAsFactors = F)
str(train_data)
summary(train_data)

convert_to_factors <- c(1, 2)
train_data[, convert_to_factors] <- data.frame(apply(train_data[convert_to_factors], 2,
                                                     as.factor))

set.seed(42)
boruta.train <- Boruta(target~.-card_id, data = train_data, doTrace = 2)