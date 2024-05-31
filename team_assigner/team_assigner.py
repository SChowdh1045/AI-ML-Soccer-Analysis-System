from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):

        # Key: team_id (1 or 2)
        # Value: RGB value (1D array) of the team color
        self.team_colors = {}
        
        # Key: player_id (any integer)
        # Value: team_id (1 or 2)
        self.player_team_dict = {}


    ############################## Main Functions (CALLED IN main.py) ##############################
    def assign_team_color(self, frame, player_frame_dict):
        player_colors = []
        
        for _, player_detection in player_frame_dict.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame, bbox) # Returns the RGB value (1D array) of the cluster center that corresponds to the player cluster.
            player_colors.append(player_color)
        
        
        # The KMeans clustering here is used to find the two distinct team colors. Even though get_player_color() returns the color of an individual player, there might be slight variations in the colors of different players on the same team due to lighting conditions, shadows, etc.
        # By clustering the player colors into two groups, we can find the average color for each team, which should be a good representation of the team's color.
        # As for the n_init parameter, it controls the number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. Inertia here is the sum of squared distances of samples to their closest cluster center. The lower the inertia, the better the clustering.
        # Setting n_init=1 for individual players means that the k-means algorithm is run only once per player, which might be a reasonable choice if the colors of individual players are not expected to vary much.
        # On the other hand, setting n_init=10 for the team colors means that the k-means algorithm is run 10 times with different centroid seeds, and the best output is chosen. This might be done to get a more robust estimate of the team colors, which are to be used in subsequent processing steps. The higher value of n_init increases the chances of finding a solution that is closer to the global optimum.
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # To be used in 'get_player_team()' function
        self.kmeans = kmeans

        # The 1 and 2 in the team_colors dictionary are keys, not indices. I put '1' and '2' as keys instead of 0 and 1 because it makes more sense to say "team 1" and "team 2" rather than "team 0" and "team 1".
        # The keys 1 and 2 are arbitrary and don't have to match the cluster labels. They are chosen to match some other part of the code where teams are referred to as 1 and 2 (in main.py). The important part is that each team color is stored somewhere in the team_colors dictionary, not necessarily which key it's stored under.
        # The cluster_centers_ attribute of the KMeans object here contains the RGB values of the cluster centers. These are the average colors of the players in each cluster, which represent the team colors.
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]



    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox) # Returns the RGB value (1D array) of the cluster center that corresponds to the player cluster.

        # The zero-index at the end is being applied to the OUTPUT of the predict function, NOT the input.
        # Here's what's happening step by step:
        # 1) player_color.reshape(1, -1) reshapes the 1D array player_color into a 2D array with one row. This is done because the predict function expects a 2D array as input.
        # 2) self.kmeans.predict(player_color.reshape(1, -1)) applies the predict function to this 2D array. The predict function returns a 1D array of predicted cluster labels. Since there's only one sample (one row in the input array), the output array contains only one element.
        # 3) The [0] at the end indexes into this output array to get the first (and only) element. This is the predicted cluster label for the player color, which is then stored in team_id.
        # So, to answer your question, the predict function is taking player_color.reshape(1, -1) as input, and the zero-index is applied to the output of the predict function.
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1 # team_id is initially 0 or 1 (the cluster labels), so this line changes it to 1 or 2 (the team IDs)

        self.player_team_dict[player_id] = team_id

        return team_id



    ############################## Helper Functions (CALLED ONLY WITHIN THIS CLASS) ##############################    
    def get_player_color(self, frame, bbox):
        # A frame/image in a video is represented as a 3D array.
        # If you have a frame/image of size 480x640, the corresponding 3D array would have a shape of (480, 640, 3), where 480 is the height, 640 is the width, and 3 is the number of color channels (RGB).
        # Its structure would look something like this (I'm doing (H-1) and (W-1) because the indices start from 0): 
        # image = [
        #           [ [R00, G00, B00], [R01, G01, B01], ..., [R0(W-1), G0(W-1), B0(W-1)] ],
        #           [ [R10, G10, B10], [R11, G11, B11], ..., [R1(W-1), G1(W-1), B1(W-1)] ],
        #           ...
        #           [ [R(H-1)0, G(H-1)0, B(H-1)0], [R(H-1)1, G(H-1)1, B(H-1)1], ..., [R(H-1)(W-1), G(H-1)(W-1), B(H-1)(W-1)] ]
        #         ]
        #
        # 'frame' is a 3D array and we are slicing it to get a smaller 3D array (cropped image).
        # So if it was frame[100:150, 200:300], the first argument is the height range (rows 100 to 150) and the second argument is the width range (columns 200 to 300).
        # The color channels are not affected by this slicing operation.
        # If I wanted to slice the RGB channels/arrays, I would have to do something like frame[100:150, 200:300, 0:2] to get the first two color channels (R and G). Adding another argument would give me an error.
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # 'cropped_image.shape[0]' gives the height of the cropped image (number of rows). Dividing it by 2 to get the top half of the image.
        # The second argument ':' means all the columns.
        # 'top_half_image' is still a 3D array.
        top_half_image = cropped_image[0:int(cropped_image.shape[0]/2), :]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # The 'kmeans.labels_' attribute returns an array where each of its element is the label/cluster index for each data point. In the context of your image clustering, each data point corresponds to a pixel in the image_2d array (from the get_player_color() function).
        # The labels/cluster indexes are integers from 0 to n_clusters-1. In this case, since n_clusters=2, the labels will be either 0 or 1.
        # For example, if 'kmeans.labels_' returns [0, 1, 0, 1, 1, 0, ...], this means that the first pixel in image_2d was assigned to cluster 0, the second pixel was assigned to cluster 1, the third pixel was assigned to cluster 0, and so on.
        # These labels/cluster indexes indicate which of the two clusters each pixel in the image belongs to, based on the K-means clustering. 
        labels = kmeans.labels_

        # 'clustered_image' is a 2D array with the same dimensions as top_half_image (height/rows and width/columns). The RGB array of each pixel in 'top_half_image' is assigned to a cluster (0 or 1) based on the K-means clustering.
        # The reshape() function is used to transform the 1D labels array back into the original 2D shape of the 'top_half_image'. Each value in 'clustered_image' corresponds to the cluster assignment (0 or 1) of the corresponding pixel in 'top_half_image'
        # So clustered_image will have the following structure (the 0s and 1s are just placeholders):
        # clustered_image = [
        #                       [0, 1, 0, 0, 1, 1, 0, ...],  # row 1
        #                       [1, 1, 0, 1, 0, 0, 1, ...],  # row 2
        #                       [0, 0, 1, 1, 0, 1, 1, ...],  # row 3
        #                       ...
        #                       [1, 0, 1, 0, 1, 0, 0, ...]   # row N
        #                   ]   
        #
        # N is the number of rows in 'top_half_image'.
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # In Python, you can index multi-dimensional arrays (like 2D or 3D arrays) by specifying an index for each dimension, separated by commas. This is known as multi-dimensional indexing.
        # Also in Python, indexing an array with 0 gives the first element and -1 gives the last element.
        # So for a 2D array like 'clustered_image', the first index corresponds to the row and the second index corresponds to the column.
        # Therefore, clustered_image[0,0] gives the top-left corner of the image, clustered_image[0,-1] gives the top-right corner, clustered_image[-1,0] gives the bottom-left corner, and clustered_image[-1,-1] gives the bottom-right corner.
        # 'corner_clusters' is an array of 4 values. Each value is the cluster assignment (0 or 1) of the corresponding corner pixel in clustered_image.
        # So, for example, if 'corner_clusters' is [0, 0, 1, 0], this means that the top-left, top-right, and bottom-right corners of the image are assigned to cluster 0, and the bottom-left corner is assigned to cluster 1.
        # These values are used to determine the most common cluster in the corners, which is assumed to be the background (non-player) cluster.
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        
        # set(corner_clusters) turns 'corner_clusters' into a set, which is a collection of unique elements. So if corner_clusters is [0, 0, 1, 0], set(corner_clusters) would be {0, 1}
        # max() is a built-in Python function that returns the largest item in an iterable or the largest of two or more arguments. The key parameter is optional and expects a function to be passed to it.
        # Here's how it works:
        # 1) max() iterates over each element in set(corner_clusters), which are the unique values in corner_clusters (0 and 1 in this case).
        # 2) For each unique value, it calls the function passed to the key parameter (corner_clusters.count). This function returns the number of times the unique value appears in the ORIGINAL corner_clusters.
        # 3) max() then compares these counts to find the maximum. The unique value that has the maximum count is returned.
        # So, even though max() is operating on set(corner_clusters), the corner_clusters.count function passed to 'key' is considering the counts of these unique values (0 and 1) in the original corner_clusters list.
        # And the reason corner_clusters.count doesn't have parenthesis after it is because I'm calling it by reference, not executing it. I'm passing the function itself to max() so that max() can call it internally.
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        
        # The player cluster is the cluster that is NOT the background (non-player) cluster. Since there are only two clusters (0 and 1), the player cluster is the one that is different from the non-player cluster.
        # For example, if the non-player cluster is 0, then the player cluster is 1-0 = 1. If the non-player cluster is 1, then the player cluster is 1-1 = 0.
        player_cluster = 1 - non_player_cluster

        # kmeans.cluster_centers_ is a 2D numpy array where each row corresponds to a cluster center. The index of the row corresponds to the cluster label.
        # The KMeans object assigns cluster labels in ascending order from 0 to k-1, where k is the number of clusters. This is the standard behavior of the sklearn.cluster.KMeans class in Python.
        # So, if you have two clusters (k=2), the labels will be 0 and 1. kmeans.cluster_centers_[0] will give the center of the first cluster (label 0), and kmeans.cluster_centers_[1] will give the center of the second cluster (label 1).
        # This makes it easy to retrieve the center of a particular cluster by using its label as an index into kmeans.cluster_centers_.
        # kmeans.cluster_centers_[x] would return a 1D array containing the RGB values of the specified cluster center.
        # For example, if you have an RGB image and you've used KMeans to cluster the pixels into two groups, kmeans.cluster_centers_ might look something like this:
        # kmeans.cluster_centers_ = [
        #                               [34, 64, 96],
        #                               [240, 248, 255]
        #                           ]
        #
        # In this case, kmeans.cluster_centers_[0] would return the array [34, 64, 96], which represents the RGB values of the first cluster center, and kmeans.cluster_centers_[1] would return the array [240, 248, 255], which represents the RGB values of the second cluster center.
        player_color = kmeans.cluster_centers_[player_cluster]

        # Returns the RGB value (1D array) of the cluster center that corresponds to the player cluster.
        return player_color


    def get_clustering_model(self, image):
        # The reshape function in Python changes the shape (i.e., the number of elements in each dimension) of an array without changing its data.
        # In this case, image.reshape(-1,3) is used to flatten the image into a 2D array where each row represents a pixel and each column represents a color channel.
        # The -1 in the reshape function is a placeholder that means "infer this dimension from the other dimension(s)". In this case, it's saying "make this dimension as large as needed to accommodate all the data given the size of the other dimension(s)".
        # The 3 means that the second dimension should have a size of 3, which corresponds to the three color channels (R, G, B).
        # So, image.reshape(-1,3) will transform the 3D array into a 2D array where each row is a pixel (R, G, B) from the cropped image. The number of rows in this 2D array will be equal to the total number of pixels in the cropped image (Height x Width)
        # For example, if the cropped image array has a shape of (100, 150, 3), the reshaped image_2d array will have a shape of (15000, 3), because there are 15000 pixels in the cropped image (100 height x 150 width = 15000 pixels).
        # So image_2d will have the following structure:
        # image_2d = [
        #               [R0, G0, B0],
        #               [R1, G1, B1],
        #               ...
        #               [R14999, G14999, B14999]
        #            ]
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters.
        # init="k-means++" is a smart initialization technique that selects initial cluster centers in a way that speeds up convergence.
        # n_init=1 determines the number of time the k-means algorithm will be run with different centroid seeds. 
        # If n_init was set to 5, the K-means algorithm would run 5 times from the beginning with different initial centroids each time. This is useful because K-means can get stuck in local minima, so running it multiple times with different initializations can help find the best solution.
        # In this case, n_init=1 is used to speed up the clustering process.
        # This line is just initializing the KMeans object with specific parameters. No computation is done right now.
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        

        # This part is where the K-means algorithm is actually run on the image data.
        # This updates the 'kmeans' object with the results of the clustering, including the final centroid locations and the assignments of data points to clusters.
        # The K-means clustering algorithm treats each pixel in the image as a point in a three-dimensional color space (RGB) (i.e. the 3-columned arrays of R,G,B values from 'image_2d'). Here's a step-by-step explanation:
        #
        # Initialization: K-means starts by randomly initializing two centroids in the RGB color space. These centroids are vectors of three elements (R, G, B), just like the pixels.
        # Assignment: Each pixel in the image is assigned to the centroid that it is closest to. The distance is typically calculated using the Euclidean distance in the RGB color space. This step effectively partitions the pixels into two clusters.
        # Update: The centroids are recalculated as the mean RGB values of all pixels in each cluster.
        # Iteration: Steps 2 and 3 are repeated until the centroids do not change significantly or a maximum number of iterations is reached.
        # The result of this process is that the pixels in the image are divided into two groups (clusters) based on their color similarity. Each cluster is represented by its centroid, which can be thought of as the "average color" of all the pixels in that cluster.
        # This is a way of reducing the color palette of the image to just two colors, which is what I want for the team assignments.
        kmeans.fit(image_2d)

        # Returns the 'kmeans' object, which contains the clustering results.
        return kmeans