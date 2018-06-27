import re
import traceback
import numpy as np


class PageRankNode(object):
    def __init__(self, id, author, title, author_num, venue, index):
        self.id = id
        self.author = author
        self.title = title
        self.author_num = author_num
        self.venue = venue
        self.index = index


def get_data_meta():
    metadata_file = "../data/aan/release/2011/acl-metadata.txt"

    with open(metadata_file, "rb") as fi:
        text = fi.read()

    id_list = re.findall(b"id = {(.*?)}", text)
    author_list = re.findall(b"author = {(.*?)}", text)
    title_list = re.findall(b"title = {(.*?)}", text)
    venue_list = re.findall(b"venue = {(.*?)}", text)

    assert len(id_list) == len(author_list) == len(title_list) == len(venue_list)

    author_index_dict = {}
    index_author_dict = {}
    author_index = -1

    venue_index_dict = {}
    index_venue_dict = {}
    venue_index = -1

    page_rank_node_dict = {}

    for i in range(len(id_list)):
        venue_format = venue_list[i].strip()
        if venue_format not in venue_index_dict:
            venue_index += 1
            venue_index_dict[venue_format] = venue_index
            index_venue_dict[venue_index] = venue_format

        for author in author_list[i].split(b";"):
            author_format = author.strip()  # .replace(b", ", b"_")
            if author_format not in author_index_dict:
                author_index += 1
                author_index_dict[author_format] = author_index
                index_author_dict[author_index] = author_format

        page_rank_node_dict[id_list[i]] = PageRankNode(id=id_list[i].strip(),
                                                       author=[author_index_dict[a.strip()] for a in
                                                               author_list[i].split(b";")],
                                                       title=title_list[i],
                                                       author_num=len(author_list[i].split(b";")),
                                                       venue=venue_index_dict[venue_list[i].strip()],
                                                       index=i)

    return page_rank_node_dict, author_index_dict, index_author_dict, venue_index_dict, index_venue_dict


def get_data_dependence(page_rank_node_dict, author_index_dict, venue_index_dict):
    page_rank_node_num = len(page_rank_node_dict)
    author_num = len(author_index_dict)
    venue_num = len(venue_index_dict)

    id_mat = np.zeros((page_rank_node_num, page_rank_node_num))
    author_mat = np.zeros((author_num, author_num))
    venue_mat = np.zeros((venue_num, venue_num))

    dependata_file = "../data/aan/release/2011/acl.txt"
    with open(dependata_file, "rb") as fi:
        text = fi.read()

    for dep in text.split(b"\n"):
        try:
            dep_format = dep.strip()
            dep_a_id, dep_b_id = dep_format.split(b" ==> ")

            dep_a_id_node = page_rank_node_dict[dep_a_id]
            dep_b_id_node = page_rank_node_dict[dep_b_id]

            dep_a_id_index = dep_a_id_node.index
            dep_b_id_index = dep_b_id_node.index

            id_mat[dep_a_id_index, dep_b_id_index] = 1

            for i in range(dep_a_id_node.author_num):
                for j in range(dep_b_id_node.author_num):
                    author_mat[dep_a_id_node.author[i], dep_b_id_node.author[j]] = 1

            venue_mat[dep_a_id_node.venue, dep_b_id_node.venue] = 1
        except Exception as e:
            # traceback.print_exc()
            print("Not found {0}".format(e))

    return id_mat, author_mat, venue_mat


def page_rank(mat,
              bias_threshold=1e-6,
              max_iter=500,
              alpha=0.85,
              epsi=0.0000000001):
    shape = mat.shape[0]
    rank = np.ones((shape, 1))
    average_factor = 1. / shape
    mat_sum = np.sum(mat, 1) + epsi
    mat_format = mat.T / mat_sum

    iter_num = 0

    while True:
        new_rank = np.matmul(mat_format, rank) * alpha + (1 - alpha) * average_factor
        bias = np.sum(np.abs(new_rank - rank)) / shape
        print("Bias: {0}".format(bias))
        if bias < bias_threshold:
            break
        iter_num += 1
        if iter_num > max_iter:
            break
        rank = new_rank

    print("Iter: {0}".format(iter_num))

    return rank


if __name__ == '__main__':
    page_rank_node_dict, author_index_dict, index_author_dict, venue_index_dict, index_venue_dict = get_data_meta()

    id_mat, author_mat, venue_mat = get_data_dependence(page_rank_node_dict,
                                                        author_index_dict,
                                                        venue_index_dict)

    id_rank = page_rank(id_mat)
    author_rank = page_rank(author_mat)
    venue_rank = page_rank(venue_mat)

    a = 1
