#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Data on kaggle. "Airline Passenger Satisfaction"
# https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction
# License CC0: Public Domain

df = pd.read_csv("airline_passenger_satisfaction.csv")


# # Data Cleaning

# In[3]:


# Set pandas options to display all columns
pd.set_option('display.max_columns', None)
# Drop the ID column
df = df.drop('ID', axis=1)
df.dropna(inplace = True)

# Saving cleaned df
original_df = df.copy()


# # Create dummy numbers for non-numeric columns

# In[4]:


# dummies:
df = pd.get_dummies(data = df, columns = ["Gender", "Customer Type", 
                                          "Type of Travel", "Class",
                                         "Satisfaction"], drop_first = True)


# In[5]:


# change True or False to 1 or 0
# Define the columns to convert
columns_to_convert = ['Gender_Male','Customer Type_Returning', 'Type of Travel_Personal', 'Class_Economy',
       'Class_Economy Plus', 'Satisfaction_Satisfied']
# Convert 'True' and 'False' to 1 and 0 in the specified columns
df[columns_to_convert] = df[columns_to_convert].replace({True: 1, False: 0})


# # Scale the data

# In[6]:


import numpy as np
from sklearn.preprocessing import StandardScaler


# In[7]:


# Specify which columns to scale
columns_to_scale = ['Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay',
       'Departure and Arrival Time Convenience', 'Ease of Online Booking',
       'Check-in Service', 'Online Boarding', 'Gate Location',
       'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness',
       'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
       'In-flight Entertainment', 'Baggage Handling']


# In[8]:


# Initialize the StandardScaler
scaler = StandardScaler()
# Creating scaled df
scale_df = df.copy()
# Fit and transform the selected columns
scale_df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


# # PCA

# In[9]:


from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


# In[10]:


decomp = PCA()
pipe = make_pipeline(decomp)
pipe.fit(scale_df)


# In[11]:


# Visualization of PCA
explained_variance = pipe['pca'].explained_variance_ratio_.cumsum()
index = [i+1 for i in range(len(explained_variance))]

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=index, y=explained_variance)
sns.scatterplot(x=index, y=explained_variance, s=100)
plt.xlim((1-0.2, len(explained_variance)+0.2))
plt.ylim((0, 1.1))
x_s, x_e = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(x_s+0.2, x_e))
ax.hlines(y=0.9, xmin=1, xmax=len(explained_variance), color='gray', linestyle='--')
plt.ylabel('Total Explained Variance Ratio')
plt.xlabel('PC')
plt.show()


# In[12]:


decomp = PCA(n_components =0.9)
pipe = make_pipeline(decomp)
pipe.fit(scale_df)


# #  KMeans Clustering

# In[13]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# In[14]:


cluster = KElbowVisualizer(KMeans(n_init=10))
pipe = make_pipeline(decomp, cluster)
pipe.fit(scale_df)
pipe[1].show()


# In[15]:


# Above suggested clusters number 5
cluster = KMeans(n_clusters=5)
decomp = PCA(n_components=0.9) 
pipe = make_pipeline(decomp, cluster)
pipe.fit(scale_df)


# #  Update df with Cluseters

# In[16]:


values = pipe[:1].transform(scale_df)
pca_labels = [f'PC{idx+1}' for idx, i in enumerate(values.T)]
scale_df = scale_df.join(pd.DataFrame(values, columns=pca_labels))
df['clusters'] = pipe['kmeans'].labels_
original_df['clusters'] = pipe['kmeans'].labels_


# # Analyzation by Visualization

# In[17]:


import math


# In[18]:


# Set a palette for colors, font and label for clusters
custom_palette = ["#ca0020","#f4a582","#f7f7f7","#92c5de","#0571b0"]
sns.set(palette = custom_palette, font="Helvetica")
cluster_names = {0:"On a \nBudget", 3:"Trouble-\nExperienced", 4:"Flight-\nDelayed",
               2:"Tech-Assist \nTarget", 1:"Satisfied \nBusiness"}
# Define the desired order of clusters
desired_cluster_order = [0, 3, 4, 2, 1] 


# In[19]:


# Create a copy of the original DataFrame
df_copy = df.copy()

# Convert the 'clusters' column to a categorical data type with the desired order
df_copy['clusters'] = pd.Categorical(df_copy['clusters'], categories=desired_cluster_order, ordered=True)

# Sort the DataFrame based on the categorical order
df_copy = df_copy.sort_values('clusters')


# In[20]:


# List of variables
var_list = ["Gender_Male", "Customer Type_Returning", "Type of Travel_Personal",
            "Class_Economy", "Class_Economy Plus", "Satisfaction_Satisfied"]

# Calculate the number of rows and columns for subplots
rows = math.ceil(len(var_list) / 3)
cols = 3

# Create subplots
fig, axs = plt.subplots(rows, cols, figsize=(20, 5 * rows))

# Iterate through the variables in var_list
for i, ax in zip(var_list, axs.flat):
    g = sns.barplot(data=df_copy, x='clusters', y=i, ax=ax, edgecolor="black")

    # Set custom x-values and x-labels based on cluster_names
    x_values = list(range(len(desired_cluster_order))) 
    custom_x_labels = [cluster_names.get(cluster, '') for cluster in desired_cluster_order]
    
    ax.set(xlabel='', ylabel='')
    ax.set_title(i, fontsize=27)
    ax.set_xticks(x_values)
    ax.set_xticklabels(custom_x_labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

# Turn off any remaining empty subplots
for i in axs.flat[::-1][:rows * cols - len(var_list)]:
    i.set_axis_off()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# In[21]:


custom_palette = [(202/255, 0, 32/255, 0.7), (5/255, 113/255, 176/255, 0.6)]

# Create subplots for each cluster
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
# Adjust the spacing between subplots using subplots_adjust
plt.subplots_adjust(wspace=-1) 

# Loop through each cluster and create a pie chart
for i, cluster in enumerate(desired_cluster_order):
    subset = df[df['clusters'] == cluster]
    satisfied_count = subset['Satisfaction_Satisfied'].sum()
    dissatisfied_count = len(subset) - satisfied_count
    
    counts = [satisfied_count, dissatisfied_count]
    
    axes[i].pie(counts, autopct='%1.1f%%', startangle=90,
               colors= custom_palette,
                labeldistance=0, textprops={'fontsize': 13})
    axes[i].set_title(cluster_names[int(cluster)],fontsize= 16)
    
plt.suptitle('',
            fontsize=25)
axes[i].legend(['Satisfied', 'Neutral or\nDissatisfied'], loc='upper center', bbox_to_anchor=(-0.5, -0), fontsize=15)

plt.tight_layout()
# plt.savefig('satisfy6.png', format='png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# In[22]:


var_list = ['Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay',
       'Departure and Arrival Time Convenience', 'Ease of Online Booking',
       'Check-in Service', 'Online Boarding', 'Gate Location',
       'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness',
       'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
       'In-flight Entertainment', 'Baggage Handling']

rows = math.ceil(len(var_list)/3)

# Create a copy of the original DataFrame
df_original_copy = original_df.copy()

# Convert the 'clusters' column to a categorical data type with the desired order
df_original_copy['clusters'] = pd.Categorical(df_original_copy['clusters'], categories=desired_cluster_order, ordered=True)

# Sort the DataFrame based on the categorical order
df_original_copy = df_original_copy.sort_values('clusters')

fig, axs = plt.subplots(rows, 3, figsize=(20, 5*rows))

for i, ax in zip(var_list, axs.flat):
    g = sns.barplot(data=df_original_copy, x='clusters', y=i, ax=ax, edgecolor="black")
    ax.set(xlabel='', ylabel='', title=i)

for i in axs.flat[::-1][:rows*3-len(var_list)]:
    i.set_axis_off()
plt.show()


# In[24]:


# Define the class categories
class_categories = ['Economy', 'Economy Plus', 'Business']
custom_labels = ['Economy', 'Economy\nPlus', 'Business']
custom_palette = ["#3288bd","#91cf60","#f1a340"]

# Create subplots for each cluster
fig, axes = plt.subplots(1, 5, figsize=(15, 4))

# Loop through each cluster and create a bar chart
for i, cluster in enumerate(desired_cluster_order):
    subset = original_df[original_df['clusters'] == cluster]
    
    # Count the number of each class category in the subset
    class_counts = [subset[subset['Class'] == category].shape[0] for category in class_categories]
    
    total_count = sum(class_counts)
    
    # Calculate the percentages
    percentages = [count / total_count * 100 for count in class_counts]
    
    axes[i].bar(class_categories, percentages, color=custom_palette,
               edgecolor="black")
    axes[i].set_title(cluster_names[int(cluster)], fontsize=15)
    axes[i].set_ylim(0, 100)
    axes[i].set_xticklabels(custom_labels, fontsize=12)

    
plt.suptitle('Percentage of Flight Class', fontsize=20)
plt.tight_layout()
plt.show()


# #  Map Visualization

# In[ ]:


import folium
from geopy.distance import geodesic

# Coordinates for New York City (latitude, longitude)
nyc_coords = (40.7128, -74.0060)

# Coordinates for Minneapolis (latitude, longitude)
minneapolis_coords = (44.9778, -93.2650)

# Coordinates for Denver (latitude, longitude)
denver_coords = (39.7392, -104.9903)

# Calculate the distance between New York City and Minneapolis
nyc_to_minneapolis_distance = geodesic(nyc_coords, minneapolis_coords).miles

# Calculate the distance between New York City and Denver
nyc_to_denver_distance = geodesic(nyc_coords, denver_coords).miles

# Create a map centered on New York City
m = folium.Map(location=nyc_coords, zoom_start=6, tiles = "stamenterrain")

# Add markers for New York City, Minneapolis, and Denver
folium.Marker(nyc_coords, tooltip='New York City').add_to(m)
folium.Marker(minneapolis_coords, tooltip='Minneapolis').add_to(m)
folium.Marker(denver_coords, tooltip='Denver').add_to(m)

# Create lines connecting the markers to represent the distances
folium.PolyLine([nyc_coords, minneapolis_coords], color='blue', weight=2.5, opacity=1).add_to(m)
folium.PolyLine([nyc_coords, denver_coords], color='red', weight=2.5, opacity=1).add_to(m)

# Add popups displaying the calculated distances
popup_minneapolis = folium.Popup(f"Distance to Minneapolis: {nyc_to_minneapolis_distance:.2f} miles", max_width=300)
popup_minneapolis.add_to(folium.Marker(location=minneapolis_coords))

popup_denver = folium.Popup(f"Distance to Denver: {nyc_to_denver_distance:.2f} miles", max_width=300)
popup_denver.add_to(folium.Marker(location=denver_coords))

# Add text labels near the lines
folium.Marker([45, -83], icon=folium.DivIcon(html='<div style="color:blue; font-weight:bold">800mil 2h45m</div>')).add_to(m)
folium.Marker([42, -96], icon=folium.DivIcon(html='<div style="color:#ff5733; font-weight:bold" >1400mil 3h45m </div>')).add_to(m)

# Save the map as an HTML file
m.save('distance_map.html')

