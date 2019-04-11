from libpysal.weights.util import get_points_array_from_shapefile
from libpysal.io.fileio import FileIO as psopen
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial import KDTree
import time

# use spatial distance to select potential candidates for clustering


def local_area_clustering(filePath, attrsName, extensiveAttrName, extensive_threshold, cluster_threshold):
    dbfReader = psopen(filePath)
    # get attribute from all cell
    attrs = np.array(dbfReader.by_col(attrsName[0]))

    extensiveAttr = np.array(dbfReader.by_col(extensiveAttrName))

    # get coordinates for all points
    spatial_attr = get_points_array_from_shapefile(filePath.split('.')[0] + '.shp')
    all_points_size = len(spatial_attr)
    left_point_number = all_points_size
    index_spatial_attr = np.arange(all_points_size)
    c = 0
    cluster_all = []
    region_list = []
    enclave_list = []
    labels = np.zeros(all_points_size, dtype=int)
    count_enclave = 0
    attr_dist_within_region = []
    # 0.set unassigned_index_array to index_spatial_attr.
    unassigned_index_list = index_spatial_attr
    t1 = time.process_time()
    obj_file = open("result.txt", "w")
    # 1.randomly select one points form unassigned_index_array, named after nearest_point_index, stored point's index
    #   to an array named cluster_array.
    # 1.1 [invalid]determine if unsaigned points > cluster_threshold, false-> 9.
    # 1.2 [revised]: After select a random point, determine the nearest 1000 points exist. We can make this point
    #               the root point of cluster. Otherwise, we put this point in to the enclaves_list. However, this point
    #               still exist in unassigned_index_list.So that other root point could use this point as well.

    while left_point_number != 0:
        print("left_point_number:", left_point_number)
        print("unassigned_index_list", len(unassigned_index_list))
        print("enclave_list:", enclave_list)
        nearest_point_index = random.choice(list(set(unassigned_index_list) - set(enclave_list)))
        tree = KDTree(spatial_attr)
        pts = np.array(spatial_attr[nearest_point_index])
        temp_index_list = tree.query(pts, cluster_threshold+1)[1].tolist()
        temp_index_list = temp_index_list[1:]
        print("pts:", nearest_point_index)
        print("temp index list:", temp_index_list)
        #print("len of temp index list(before):", len(temp_index_list))
        # 3/27/2019
        select_list = [labels[i] for i in temp_index_list]
        qualified_number = len(temp_index_list) - np.count_nonzero(select_list)

        # for cluster in cluster_all:
        #    temp_index_list = list(set(temp_index_list).difference(set(cluster)))
        if count_enclave > (all_points_size * 0.01):
            enclave_list = unassigned_index_list
            break
        # revised 3/27/2019
        if qualified_number < (cluster_threshold * 0.4):
            count_enclave += 1
            enclave_list.append(nearest_point_index)
            left_point_number = len(list(set(unassigned_index_list) - set(enclave_list)))
            continue

        # 2.[invalid]determine cluster_array < min_threshold. yse->jump to 3, no -> jump to 8
        #    while(len(cluster_list) <= extensive_threshold
        #     and len(temp_unassigned_list) > (extensive_threshold - len(cluster_list))):
        # 2. when there exist points in unassigned_index_list, try to set up cluster. Determine if
        c += 1
        attr_dist_within_region.append(0)
        cluster_list, attr_dist_within_region[c-1] = clustering(nearest_point_index, temp_index_list, attrs,
                                                              extensive_threshold, spatial_attr, tree,
                                                              cluster_all, cluster_threshold)
        for index in cluster_list:
            labels[index] = c
        # 3.find k nearest neighbors for nearest_point_index, return their index, stored them in temp_index_array.
        '''
            pts = np.array(spatial_attr[nearest_point_index])
            temp_index_list = tree.query(pts, area_size)[1].tolist()
            for cluster in cluster_all:
                temp_index_list = list(set(temp_index_list).difference(set(cluster)))
            # 4.find their common points use the index, compare between unassigned_index_array with temp_index_array,
            #   restore the unassigned_index_array.
            temp_unassigned_list = list(set(temp_unassigned_list).intersection(temp_index_list))
            # print(len(temp_unassigned_list))
            # 5.find nearest point to points from cluster_list amoung points we stored in unsaigned_index_array.
            #   and named as nearest_point_index.(use attribute distance), add nearest_point_index to cluster_array.
            cluster_list, attr_dist_within_region, nearest_point_index = \
                add_attribute_nearest_point(temp_unassigned_list, attrs, cluster_list, attr_dist_within_region)
        '''
        # 6.delete the point from unassigned_index_array.
        print("cluster list:", cluster_list)
        unassigned_index_list = list(set(unassigned_index_list).difference(set(cluster_list)))
        cluster_all.append(cluster_list)
        print("cluster all:", len(cluster_all))
        # 7.jump to 2
        region_list.append(c)
    # 9.situation that unassigned_index_list is not enough for generating another cluster:
    while len(enclave_list):
        # 10. for point in unassigned_index_list, use its' attr compare with every point in each cluster.
        for unass_point in enclave_list:
            print("len of enclave list:", len(enclave_list))
            min_dist_attr = float('inf')
            temp_cluster = 0
            count = 0
            # 11. find the closed cluster, add it index to cluster_list.
            for cluster in cluster_all:
                dist_attr = 0
                for point_in_cluster in cluster:
                    dist_attr += abs((attrs[unass_point] - attrs[point_in_cluster]))
                if dist_attr < min_dist_attr:
                    min_dist_attr = dist_attr
                    temp_cluster = count
                count += 1
            attr_dist_within_region[temp_cluster] += min_dist_attr
            cluster_all[temp_cluster].append(unass_point)
            labels[unass_point] = temp_cluster
            enclave_list.remove(unass_point)
    len_cluster = []
    t2 = time.process_time()
    for cluster in cluster_all:
        len_cluster.append(len(cluster))
    print("The number of regions is", len(region_list))
    print("The region extensive attribute is", len_cluster)
    print("Attribute distance is", attr_dist_within_region)
    print("Total Attribute distance is", sum(attr_dist_within_region))
    print("Time cost:", (t2-t1))
    obj_file.write("The number of regions is %d\n" %len(region_list))
    # obj_file.write("The region extensive attribute is %f\n" %(len_cluster))
    obj_file.write("Attribute distance is:%f\n" %sum(attr_dist_within_region))
    obj_file.write("Time cost: %f\n" %(t2-t1))
    obj_file.close()
    #gp_shp = gp.read_file(filePath.split('.')[0] + '.shp')
    #gp_shp['regions'] = labels
    #gp_shp.plot(column='regions', legend=True)
    #plt.show()


def add_attribute_nearest_point(temp_unassigned_list, attrs, cluster_list, attr_dist_within_region):
    # 0.select attribute from attrs by unassigned_index_list, use dictionary to store{'index','attr'}
    unassigned_attr = []
    assigned_attr = []
    attr_dist = 0
    min_dist = float('inf')
    for point in temp_unassigned_list:
        unassigned_attr.append(attrs[point])
    unassigned_index_attr_dic = dict(zip(temp_unassigned_list, unassigned_attr))
    # 1.find nearest point to points from cluster_list amoung points we stored in unsaigned_index_array.
    for point in cluster_list:
        assigned_attr.append(attrs[point])
    assigned_index_attr_dic = dict(zip(cluster_list, assigned_attr))
    print("len of temp_unassigned_list", len(temp_unassigned_list))
    cluster_list.append("to_revised")
    attr_dist_within_region.append("to_revised")
    for point in unassigned_index_attr_dic:
        temp_unass_attr = unassigned_index_attr_dic[point]
        attr_dist = 0
        for point_ass in assigned_index_attr_dic:
            attr_dist += abs(temp_unass_attr - assigned_index_attr_dic[point_ass])
        if attr_dist < min_dist:
            min_dist = attr_dist
            # 2.stored that points index as attr_index, stored total value from unassigned point to
            cluster_list[-1] = point
            attr_dist_within_region[-1] = min_dist
    return cluster_list, attr_dist_within_region, cluster_list[-1]


def clustering(point_index, temp_unassigned_list, attrs, extensive_threshold, spatial_attr, tree, cluster_all,
               cluster_threshold):
    # 0. After randomly select point, and decide if it fit the condition. Put this point in assigned_list.
    #    Calculate this point to each point in temp_unassigned_list. Save it to a temp_attr_list.
    # print("len of temp unassigned_list(before):", len(temp_unassigned_list))
    cluster_list = [point_index]
    row_list = []
    temp_attr_list = []
    cluster_dist_attr = 0
    for point in temp_unassigned_list:
        temp_attr_list.append(abs(attrs[point] - attrs[point_index]))
    # 1. Set a matrix which contains the attributes between two points. Row of the matrix is the points in assigned_list
    #    The columns are the points that in temp_unassigned_list.
    matrix = np.mat(temp_attr_list)
    row_list.append(point_index)
    # 2. Sum(value in one column), compare result and select the minimum one.figure out which column it belongs to.
    #    Find out the index of point it correspond.Delete it from the Matrix column. Add it to cluster_list.
    while len(cluster_list) <= extensive_threshold:
        temp_attr_list = []
        min_attr = float('inf')
        count_column = 0
        for column in matrix.transpose()[:]:
            #print(column)
            temp_dist_attr = np.sum(column)
            #print(temp_dist_attr)
            if temp_dist_attr < min_attr:
                min_attr = temp_dist_attr
                #print("min_attr:", min_attr)
                nearest_point_column = count_column
            count_column += 1
        cluster_dist_attr += min_attr
        matrix = np.delete(matrix, nearest_point_column, 1)
        cluster_list.append(temp_unassigned_list[nearest_point_column])
        del temp_unassigned_list[nearest_point_column]
        # 2.1 Find the intersect of the new point's kd tree and temp_unassigned_list.
        nearest_point_index = temp_unassigned_list[nearest_point_column]
        pts = np.array(spatial_attr[nearest_point_index])
        temp_index_list = tree.query(pts, cluster_threshold)[1].tolist()
        # 2.2  Eliminate points that already exist in cluster.
        for cluster in cluster_all:
            temp_index_list = list(set(temp_index_list).difference(set(cluster)))
        # 2.3 Save irrelevant points into temp_intersection. Delete irrelevant point.
        temp_intersection = list(set(temp_unassigned_list).intersection(set(temp_index_list)))
        temp_intersection = list(set(temp_unassigned_list).difference(set(temp_intersection)))
        #print("len of temp unassigned_list:", len(temp_unassigned_list))
        #print("len of intersection:", len(temp_intersection))
        for index_intersection in temp_intersection:
            count_index = 0
            for index_unassigned in temp_unassigned_list:
                if index_intersection == index_unassigned:
                    matrix = np.delete(matrix, count_index, 1)
                    del temp_unassigned_list[count_index]
                count_index += 1
        # 3. Calculate this point with others points in the temp_unassigned_list. Update the temp_attr_list.
        #    Save it to a new row in the matrix. Jump to 2.
        for point in temp_unassigned_list:
            temp_attr_list.append(abs(attrs[point] - attrs[nearest_point_index]))
        temp_attr_array = [[]]
        temp_attr_array[0] = np.array(temp_attr_list)
        matrix = np.r_[matrix, temp_attr_array]
    return cluster_list, cluster_dist_attr


if __name__ != 'main':
    filePath = 'soil_precip_temp_field_projected.dbf'
    local_area_clustering(filePath, ['nccpi2cs'], 'field_ct', 100, 1500)


