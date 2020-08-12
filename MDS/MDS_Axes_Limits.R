library(reticulate)
library(bios2mds)

pickle <- import("pickle")
builtins <- import_builtins()

labels <- pickle$load(builtins$open('labels.pkl', 'rb'))
distances <- pickle$load(builtins$open('distances.pkl', 'rb'))
positions <- pickle$load(builtins$open('positions.pkl', 'rb'))

