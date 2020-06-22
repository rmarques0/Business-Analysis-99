#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


99 CHALLENGE - DATA ANALYST POSITION

CODE INTENDED TO DO THE DATA CLEANING AND EXPLORATION OF THE DATASETS

AUTHOR: Rodrigo Sousa Marques
RMARQUES.ENGINEER@GMAIL.COM
063 99230.2552



"""

# IMPORT LIBRARIES
import pandas as pd               # DataFrame support
import numpy as np                # algebra / computations

import matplotlib.pyplot as plt   # plotting
import seaborn as sns             # fancier plotting
%matplotlib inline 


# IMPORT DATA
orders_filepath = "orders.csv" #FILEPATH OF ORDERS.CSV
trips_filepath = "trips.csv"   #FILEPATH OF TRIPS.CSV



# IMPORT DATA
base_trips = pd.read_csv(trips_filepath,
                         engine='c',
                         infer_datetime_format=True, # to speed-up datetime parsing
                         parse_dates=['pickup_datetime', 'dropoff_datetime'])

base_orders = pd.read_csv(orders_filepath, 
                          engine='c',
                          infer_datetime_format=True, 
                          parse_dates=['pickup_datetime'])


## MAKING SAMPLE OF DATA TO SPEEDUP CALC
#sample_number = 2000000
#trips = base_trips.sample(sample_number)
#orders = base_orders.sample(sample_number)

trips = base_trips
orders = base_orders


# CHECK DATA USAGE
print('Memory usage trips, Mb: {:.2f}\n'.format(trips.memory_usage().sum()/2**20))
print('Memory usage orders, Mb: {:.2f}\n'.format(orders.memory_usage().sum()/2**20))


# OVERALL INFO
print('Trips Info: ---------------------')
print(trips.info())
print('Orders Info: ---------------------')
print(orders.info())


#CHECK FOR MISSING VALUES
print(trips.isnull().sum()) 
print(orders.isnull().sum()) 


# CHECK FOR DUPLICATES: NO DUPLICATES
print('No of Duplicates, trips - order_id: {}'.format(len(trips) - 
                                              len(trips.drop_duplicates(subset='order_id'))))
print('No of Duplicates, orders - order_id: {}'.format(len(orders) - 
                                              len(orders.drop_duplicates(subset='order_id'))))

# CHECK GEOGRAPHICAL BOUNDS, Latitude: 0.0 to 400.9, Longitude: -171.4 to 121.8
print('trips Latitude bounds: {} to {}'.format(
    max(trips.pickup_latitude.min(), trips.dropoff_latitude.min()),
    max(trips.pickup_latitude.max(), trips.dropoff_latitude.max())
))
print('trips Longitude bounds: {} to {}'.format(
    max(trips.pickup_longitude.min(), trips.dropoff_longitude.min()),
    max(trips.pickup_longitude.max(), trips.dropoff_longitude.max())
))

# COUNT UNIQUE DRIVERS: 
print('driver_id   count: {}'.format(len(trips.driver_id.unique())))
# COUNT UNIQUE PASSENGERS 
print('driver_id   count: {}'.format(len(trips.passenger_id.unique())))
# DATETIME RANGE - 2014-03-01 00:00:00 to 2014-05-31 18:57:00
print('Datetime range: {} to {}'.format(trips.pickup_datetime.min(), 
                                        trips.dropoff_datetime.max()))



# CALCULATE TRIP DURATION IN MINUTES
duration = (trips['dropoff_datetime'] - trips['pickup_datetime']).dt.seconds / 60
trips = trips.assign(trip_duration = duration)
print('Trip duration in minutes: {} to {}'.format(
    trips.trip_duration.min(), trips.trip_duration.max()))





## CLEANING OUTLIERS 




#DEALING WITH TRIP DURATIONS OUTLIERS
duration_outliers=np.array([False]*len(trips))
y = np.array(trips.trip_duration)

#mark outliers  consider only trip_durations btw 1 and 60 minutes
duration_outliers[y>60]=True 
duration_outliers[y<1]=True
print('There are %d entries that have trip duration too long or too short'% sum(outliers))

#total of 113771 - 1% of completed trips
trips = trips.assign(duration_outliers=duration_outliers)
#drop outliers
trips_clean = trips[duration_outliers==False]
print('There are %d entries that have trip duration too long or too short'% sum(trips_clean.duration_outliers))

y = np.array(trips_clean.trip_duration)
plt.figure(figsize=(12,5))
plt.subplot(131)
plt.plot(range(len(y)),y,'.');plt.ylabel('trip_duration');plt.xlabel('index');plt.title('val vs. index')
plt.subplot(132)
sns.boxplot(y=trips_clean.trip_duration)
plt.subplot(133)
sns.distplot(y,bins=30, color="m");plt.yticks([]);plt.xlabel('trip_duration');plt.title('trips_clean');plt.ylabel('frequency')
#plt.hist(y,bins=50);


# Remove rides from away area
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
trips_clean = trips_clean[(trips_clean.pickup_longitude> xlim[0]) & (trips_clean.pickup_longitude < xlim[1])]
trips_clean = trips_clean[(trips_clean.dropoff_longitude> xlim[0]) & (trips_clean.dropoff_longitude < xlim[1])]
trips_clean = trips_clean[(trips_clean.pickup_latitude> ylim[0]) & (trips_clean.pickup_latitude < ylim[1])]
trips_clean = trips_clean[(trips_clean.dropoff_latitude> ylim[0]) & (trips_clean.dropoff_latitude < ylim[1])]


longitude = list(trips_clean.pickup_longitude) + list(trips_clean.dropoff_longitude)
latitude = list(trips_clean.pickup_latitude) + list(trips_clean.dropoff_latitude)
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)
plt.show()


loc_trips_clean = pd.DataFrame()
loc_trips_clean['longitude'] = longitude
loc_trips_clean['latitude'] = latitude


loc_trips_clean = loc_trips_clean.sample(1000000)



#CLUSTER REGIONS BASED ON PICKUP & DROPOFF LOCATIONS
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

kmeans = KMeans(n_clusters=15, random_state=0, n_init = 5).fit(loc_trips_clean)
loc_trips_clean['label'] = kmeans.labels_


plt.figure(figsize = (10,10))
for label in loc_trips_clean.label.unique():
    plt.plot(loc_trips_clean.longitude[loc_trips_clean.label == label],loc_trips_clean.latitude[loc_trips_clean.label == label],
             '.', alpha = 0.3, markersize = 0.3)

plt.title('Clusters of New York')
plt.show()



fig,ax = plt.subplots(figsize = (10,10))
for label in loc_trips_clean.label.unique():
    ax.plot(loc_trips_clean.longitude[loc_trips_clean.label == label],loc_trips_clean.latitude[loc_trips_clean.label == label],'.', alpha = 0.4, markersize = 0.1, color = 'gray')
    ax.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')
    ax.annotate(label, (kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'b', fontsize = 20)
ax.set_title('Cluster Centers')
plt.show()



# ENCODING DATE

# DAYS OF WEEK (DOW) MAPPING
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# MONTHS MAPPING
mm_names = [
    'January', 'February', 'March', 'April']
# MONTH (pickup and dropoff)
trips_clean['mm_pickup'] = trips_clean.pickup_datetime.dt.month.astype(np.uint8)
trips_clean['mm_dropoff'] = trips_clean.dropoff_datetime.dt.month.astype(np.uint8)
# DOW
trips_clean['dow_pickup'] = trips_clean.pickup_datetime.dt.weekday.astype(np.uint8)
trips_clean['dow_dropoff'] = trips_clean.dropoff_datetime.dt.weekday.astype(np.uint8)
# DAY HOUR
trips_clean['hh_pickup'] = trips_clean.pickup_datetime.dt.hour.astype(np.uint8)
trips_clean['hh_dropoff'] = trips_clean.dropoff_datetime.dt.hour.astype(np.uint8)





#VISUALIZATIONS



# PICKUP COUNT DISTRIBUTION: HOUR OF DAY
plt.figure(figsize=(12,5))
data = trips_clean.groupby('hh_pickup').aggregate({'order_id':'count'}).reset_index()
sns.barplot(x='hh_pickup', y='order_id', data=data, palette="rocket")
plt.title('Pick-ups Hour Distribution')
plt.xlabel('Hour of Day, 0-23')
plt.ylabel('No of trips_clean made')


# PICKUP COUNT DISTRIBUTION: DOW
plt.figure(figsize=(12,5))
data = trips_clean.groupby('dow_pickup').aggregate({'order_id':'count'}).reset_index()
sns.barplot(x='dow_pickup', y='order_id', data=data, palette="rocket")
plt.title('Pick-ups Weekday Distribution')
plt.xlabel('Day of week')
plt.xticks(range(0,7), dow_names, rotation='horizontal')
plt.ylabel('No of trips_clean made')


# PICKUP COUNT DISTRIBUTION: MONTH
plt.figure(figsize=(12,5))
data = trips_clean.groupby('mm_pickup').aggregate({'order_id':'count'}).reset_index()
sns.barplot(x='mm_pickup', y='order_id', data=data, palette="rocket")
plt.title('Pick-up Month Distribution')
plt.xlabel('Month')
plt.xticks(range(0,3), mm_names[:3], rotation='horizontal')
plt.ylabel('No of trips_clean made')


# PICKUP HEATMAP: DOW X HOUR
plt.figure(figsize=(12,5))
sns.heatmap(data=pd.crosstab(trips_clean.dow_pickup, 
                             trips_clean.hh_pickup, 
                             values=trips_clean.order_id, 
                             aggfunc='count',
                             normalize='index'))
plt.title('Pickup heatmap, Day-of-Week vs. Day Hour')
plt.ylabel('Weekday') ; plt.xlabel('Day Hour, 0-23')
plt.yticks(range(0,7), dow_names[::-1], rotation='horizontal')

# PICKUP HEATMAP: MONTH X HOUR
plt.figure(figsize=(12,5))
sns.heatmap(data=pd.crosstab(trips_clean.mm_pickup, 
                             trips_clean.hh_pickup, 
                             values=trips_clean.order_id, 
                             aggfunc='count',
                             normalize='index'))
plt.title('Pickup heatmap, Month vs. Day Hour')
plt.ylabel('Month') ; plt.xlabel('Day Hour, 0-23')
plt.yticks(range(0,4), mm_names[:4][::-1], rotation='horizontal')

# PICKUP HEATMAP: MONTH X DOW
plt.figure(figsize=(12,5))
sns.heatmap(data=pd.crosstab(trips_clean.mm_pickup, 
                             trips_clean.dow_pickup, 
                             values=trips_clean.order_id, 
                             aggfunc='count',
                             normalize='index'))
plt.title('Pickup heatmap, Month vs. Day-of-Week')
plt.ylabel('Month') ; plt.xlabel('Weekday')
plt.xticks(range(0,7), dow_names, rotation='vertical')
plt.yticks(range(0,4), mm_names[:4][::-1], rotation='horizontal')



#CALC CANCELED ORDERS

print("Percentage of cancelled orders file : ", 
      ((orders['order_id'].count() - trips['order_id'].count()) / orders['order_id'].count())*100  )
print("\n") 



#DROP DATE ENCODER - NO NEED IN OUTPUT
trips_clean = trips_clean.drop(columns=['mm_pickup','mm_dropoff','dow_pickup',
                                        'dow_dropoff','hh_pickup','hh_dropoff'])




#EXPORTING DATA - TO BE USED ON THE DASHBOARDS

geo_sample = loc_trips_clean
geo_output = pd.DataFrame(geo_sample)
trips_output = pd.DataFrame(trips_clean)
orders_output = pd.DataFrame(orders)



geo_output.to_csv('geo_sample.csv', index=False)
trips_output.to_csv('strips_clean.csv', index=False)
orders_output.to_csv('sorders.csv', index=False)


