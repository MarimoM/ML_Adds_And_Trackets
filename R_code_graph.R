library(XML)
library(rvest)
library("igraph")

dir_path <- "C:/Users/Margarita/Desktop/Magistras/html_test/"
url_dataframe <- "C:/Users/Margarita/Desktop/Magistras/magistras_test.txt"
trackers_df <- read.csv(url_dataframe, sep = "'")
htmls <- list.files(dir_path)

bendri_duom <- data.frame(matrix(ncol = 5, nrow = 0))
zymiu_duom <- data.frame(matrix(ncol = 7, nrow = 0))
length(htmls)

trackers_file <- url_dataframe
trackers_df <- read.csv(trackers_file, sep = "'")
colnames(trackers_df) <- c("Domain", "Request_URL", "Request_length", "Content_type", 
                           "Class", "Content_length_ratio", "Received_cookies", 
                           "Send_cookies", "Cookie_size", "Cookie_age", "Third_party", 
                           "Subdomain", "Parameters_nr", "Parameters")
domains <- trackers_df$Domain

making_graph <- function(paths){
  g <- make_empty_graph()
  max_p <- length(paths)
  for (var in 1:length(paths[1:max_p]))
  {
    path <- paths[[var]]
    full_path <- paste0(path, collapse = '', sep = '/')
    number_of_members <- length(path)
    g <- add.vertices(g, nv = 1, label = path[length(path)], id = full_path)
    
    if (number_of_members == 2)
    {
      parent <- paste0(path[1], sep = '/', collapse = '')
      parent_vertex <- which(V(g)$id == parent)
      g <- add.edges(g, edges = c(parent_vertex[1], var))
    }
    
    if (number_of_members > 2)
    {
      num <- number_of_members-1
      parent <- paste0(path[1:num], sep = '/',collapse = '')
      parent_vertex <- which(V(g)$id == parent)
      child <- paste0(path[1:(num+1)], sep = '/',collapse = '')
      child_vertex <- which(V(g)$id == child)
      
      if (num == number_of_members-1)
      {
        child_vertex <- which(V(g)$id == full_path)
        last_child <- child_vertex[length(child_vertex)]
        last_parent <- parent_vertex[length(parent_vertex)]
        g <- add.edges(g, edges =  c(last_parent, last_child))
      } 
      else if(are.connected(g, parent_vertex, child_vertex) == F)
      {
        last_parent <- parent_vertex[length(parent_vertex)]
        last_child <- child_vertex[length(child_vertex)]
        g <- add.edges(g, edges =  c(last_parent, last_child))
      }
    }
  }
  return(g)
}

bendri_duom <- NULL
max_p <- length(htmls)

counter <- 0
for (h in htmls[1:length(htmls)]){
  counter <- counter + 1
  print(counter)
  d <- strsplit(h, "_", fixed=T)[[1]][1]
  domain <- trackers_df$Domain[grepl(d, domains, fixed = TRUE)][1]
  if (is.na(domain)){ next }
  
  p <- paste0(dir_path, h)
  parsed <- htmlParse(p, encoding="UTF-8")
  paths <- xpathSApply(parsed, "//*", function(y) paste(unlist(xmlAncestors(y, fun=xmlName))))
  
  urls <- subset(trackers_df, Domain == domain)
  a <- lapply(paths, paste0, sep = '/', collapse = '')
  r_html <- read_html(p, encoding = "UTF-8")
  
  
  href_attr <- html_attr(html_nodes(r_html, "*"), "href")
  src_attr <- html_attr(html_nodes(r_html, "*"), "src")
  
  #if(length(a) - 1 != length(href_attr)){ next } 
  skirtumas = length(unlist(a[2:length(a)])) - length(unlist(src_attr))
  
  df <- data.frame(paths = unlist(a[2:(length(a)-skirtumas)]), href = unlist(href_attr), src = unlist(src_attr))
  
  for (var in 1:length(df$src))
  {
    if (is.na(df$src[var]) == FALSE)
    {
      if (startsWith(df$src[var], '//'))
      {
        full_url <- paste0('https:', df$src[var], collapse = '')
        df$src[var] <- full_url
      }
    }
  }
  
  for (var in 1:length(df$href))
  {
    if (is.na(df$href[var]) == FALSE)
    {
      if (startsWith(df$href[var], '//'))
      {
        full_url <- paste0('https:', df$href[var], collapse = '')
        df$href[var] <- full_url
      }
    }
  }
  
  in_table <- c()
  cont_type <- c()
  for (var in 1:nrow(df))
  {
    href <- df$href[var]
    src <- df$src[var]
    if (is.na(href) == FALSE || is.na(src) == FALSE)
    {
      smth1 <- unlist(lapply(urls, function(url) url == href))
      smth2 <- unlist(lapply(urls, function(url) url == src))
      left <- na.omit(c(smth1, smth2))
      if (sum(left) > 0)
      {
        w <- which(left)
        in_table <- append(in_table, trackers_df$Class[w][1])
        cont_type <- append(cont_type, trackers_df$Content_type[w][1])
      }
      else
      {
        in_table <- append(in_table, NA)
        cont_type <- append(cont_type, NA)
      }
    }else{
      in_table <- append(in_table, NA)
      cont_type <- append(cont_type, NA)
    }
  }
  
  df$type <- in_table
  df$content_type <- cont_type
  
  #-------------------------------- MAKING A GRAPH
  g <- making_graph(paths)
  unknown <- subset(df[1:max_p,],  startsWith(href, 'https') | startsWith(src, 'https'))
  clean <- subset(df[1:max_p,], type == 'CLEAN')
  notclean <- subset(df[1:max_p,], type == 'NOTCLEAN')
  
  #-------------------------------- METRIKOS -------------------------------------
  # Bendros grafo metrikos
  edge_nr <- gsize(g)
  vertex_nr <- length(V(g))
  skersmuo <- diameter(g)
  vid_atstumas <- mean_distance(g)
  nuorodu_kiekis <- nrow(unknown)
  santykis <- vertex_nr/nuorodu_kiekis
  laipsnis_out <- degree(g, mode = 'out')
  vid_laipsnis <- sum(laipsnis_out) / vertex_nr
  
  bendri_duom_new <- data.frame(domenas <- domain, grafo_dydis = skersmuo, nuorodu_kiekis = nuorodu_kiekis,
                                virsuniu_kiekis = vertex_nr, v_n_ratio = santykis,
                                briaunos = edge_nr)
  
  bendri_duom <- rbind(bendri_duom_new, bendri_duom)
}
write.csv(bendri_duom,"C:/Users/Margarita/Desktop/Magistras/g_test_1.csv", row.names = FALSE)

# write.csv(zymiu_duom,"C:/Users/Margarita/Desktop/Grafai/nuorodos.csv", row.names = FALSE)
# write.csv(trackers_df,"C:/Users/Margarita/Desktop/Grafai/magistras_duom.csv", row.names = FALSE)
bendri_duom
