from bmp_reader import ReadBMPFile
import numpy as np
from struct import pack

# quantization matrix
Y_mat = [[8, 6, 5, 8, 12, 20, 26, 31],
         [6, 6, 7, 10, 13, 29, 30, 28],
         [7, 7, 8, 12, 20, 29, 35, 28],
         [7, 9, 11, 15, 26, 44, 40, 31],
         [9, 11, 19, 28, 34, 55, 52, 39],
         [12, 18, 28, 32, 41, 52, 57, 46],
         [25, 32, 39, 44, 52, 61, 60, 51],
         [36, 46, 48, 49, 56, 50, 52, 50],
         ]
CbCr_mat = [[9, 9, 12, 24, 50, 50, 50, 50],
            [9, 11, 13, 33, 50, 50, 50, 50],
            [12, 13, 28, 50, 50, 50, 50, 50],
            [24, 33, 50, 50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50, 50, 50, 50],
            ]
DCT_mat = np.load('./tables/DCT.npy')


def cos_mat_generator(u, v, N):
    """generate the matrix of cos by u,v"""
    ret = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(np.cos((2 * i + 1) * u * np.pi / 2 / N) * np.cos((2 * j + 1) * v * np.pi / 2 / N))
        ret.append(row)
    return np.array(ret)


def C(u, v):
    """generate the coefficient of DCT by u,v"""
    if u == 0 and v == 0:
        return 1
    if u == 0 or v == 0:
        return np.sqrt(2)
    return 2


def DCT(block, N=8):
    new_mat = np.zeros((N, N))
    for u in range(8):
        for v in range(8):
            new_mat[u, v] = np.multiply(block, DCT_mat[u, v]).sum() * C(u, v) / N
    return new_mat


def tran_2d_1d(block):
    a = len(block)
    ret = []
    length = list(range(1, 2 * a))
    for lens in length:
        for one in range(lens):
            if one > a - 1 or lens - one - 1 > a - 1:
                continue
            if lens % 2 == 1:
                ret.append(block[lens - one - 1][one])
            else:
                ret.append(block[one][lens - one - 1])
    return np.array(ret)


def REL_encode(zigzaged_list):
    counter = 0
    ret = []
    # find the last which is not equal to 0
    ends = len(zigzaged_list) - 1
    for i in list(range(len(zigzaged_list)))[::-1]:
        if zigzaged_list[i] != 0:
            ends = i
            break
        if i == 0:
            # if the first one is 0, only EOB left
            ends = 0

    for each in zigzaged_list[:ends + 1]:
        if each != 0:
            ret.append((counter, each))
            counter = 0
        else:
            counter = counter + 1
            if counter == 16:
                ret.append((15, 0))
                counter = 0
    if ends != len(zigzaged_list) - 1:
        # if the last number not equal to 0 doesn't locate the end of the list,  append the EOB
        ret.append((-1, 'EOB'))
    return ret


def encode_first(inputs):
    if inputs == 0:
        return 0, '0b'
    abs_value = abs(inputs)
    n = 0
    while abs_value != 0:
        abs_value = abs_value >> 1
        n = n + 1
    if inputs > 0:
        return n, bin(inputs)
    lists = list(bin(abs(inputs))[2:])
    out = ['0b']
    for each in lists:
        if each == '1':
            out.append('0')
        else:
            out.append('1')
    return n, ''.join(out)


def bit_encoding(pairs_input):
    """
    bit encode
    :param pairs_input: [(0,35),(0,7),(3,-6),(0,-2),(2,-9),(15,0),(2,8),(-1,'EOB')]
    :return: (0, 6, '0b100011'), (0, 3, '0b111'), (3, 3, '0b001'),
            (0, 2, '0b01'), (2, 4, '0b0110'), (15, 0, ''), (2, 4, '0b1000'), (-1, 'EOB')]
    """
    news = [(each[0], *encode_first(each[1])) if each[0] != -1 else each for each in pairs_input]
    return news


def load_file(file_path):
    ret = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            ret.append(line.split('\t\t')[1])
    return ret


class huffman_table:
    def __init__(self):
        self.AC_chr = load_file('./tables/AC_chrominance.txt')
        self.AC_lum = load_file('./tables/AC_luminance.txt')
        self.DC_chr = load_file('./tables/DC_chrominance.txt')
        self.DC_lum = load_file('./tables/DC_luminance.txt')

    def find_DC(self, recv, types):
        """
        find the correspond encoding for DC coefficient
        :param recv: (0,5,'0b11011')
        :param types: C = chrominance | L = luminance
        :return:
        """
        if types == 'C':
            num = recv[1]
            trans = self.DC_chr[num]
        else:
            num = recv[1]
            trans = self.DC_lum[num]
        return ''.join([trans, recv[2][2:]])

    def find_AC(self, recv, types):
        """
        find the correspond encoding for AC coefficient
        :param recv: (0,5,'0b11011') | (-1, 'EOB') | (2,3,'0b101'|(15,0,'0b') | (15,3,'0b111'))
        :param types: C = chrominance | L = luminance
        :return:
        """
        if types == 'C':
            if recv[0] == -1:
                return self.AC_chr[0]
            if recv[0] == 15 and recv[1] == 0:
                return self.AC_chr[10 * recv[0] + 1]
            num = 10 * recv[0] + recv[1] if recv[0] < 15 else 10 * recv[0] + recv[1] + 1
            trans = self.AC_chr[num]
        else:
            if recv[0] == -1:
                return self.AC_lum[0]
            if recv[0] == 15 and recv[1] == 0:
                return self.AC_lum[10 * recv[0] + recv[1] + 1]
            num = 10 * recv[0] + recv[1] if recv[0] < 15 else 10 * recv[0] + recv[1] + 1
            trans = self.AC_lum[num]
        return ''.join([trans, recv[2][2:]])

    def huffman_encoding_chrominance(self, input_list):
        """
        encode the chrominance part
        :param input_list: [(0,6,'0b11011'), (0, 3,'0b101'), (15, 0, ''), (2, 4), (-1, 'EOB')]
        :return: ret:['111011011', '10010110111', '1010']
        """
        DC = self.find_DC(input_list[0], types='C')
        AC = [self.find_AC(each, types='C') for each in input_list[1:]]
        AC.insert(0, DC)
        return AC

    def huffman_encoding_luminance(self, input_list):
        """
        encode the luminance part
        :param input_list: [(0,6,'0b11011'), (0, 3,'0b101'), (15, 0， ''), (2, 4), (-1, 'EOB')]
        :return: ret:['100100011', '10010110111', '1010']
        """
        DC = self.find_DC(input_list[0], types='L')
        AC = [self.find_AC(each, types='L') for each in input_list[1:]]
        AC.insert(0, DC)
        return AC


def write_file(data, h, w, jpeg_path):
    with open(jpeg_path, 'wb') as f:
        # basic information for jpeg
        identifier = 'FFD8 FFE0 0010 4A46 4946 00 01 01 00 0001 0001 0000'
        f.writelines(str_hex_to_bytes(identifier))
        # write quantization table
        Y_list = [hex(each)[2:].zfill(2) for each in tran_2d_1d(Y_mat)]
        CbCr_list = [hex(each)[2:].zfill(2) for each in tran_2d_1d(CbCr_mat)]
        quantization_table_info = ['FFDB', hex(len(Y_list) + 3)[2:].zfill(4), '00'] + Y_list
        f.writelines(str_hex_to_bytes(''.join(quantization_table_info)))
        quantization_table_info = ['FFDB', hex(len(CbCr_list) + 3)[2:].zfill(4), '01'] + CbCr_list
        f.writelines(str_hex_to_bytes(''.join(quantization_table_info)))
        # picture information
        pic_info = 'FFC0 0011 08' + hex(h)[2:].zfill(4) + hex(w)[2:].zfill(4) + '03' + '01 11 00 02 11 01 03 11 01'
        f.writelines(str_hex_to_bytes(pic_info))
        # write the huffman table
        DC_luminance = solve_huffman_table('DC_luminance')
        f.writelines(str_hex_to_bytes('FFC4 001F 00' + DC_luminance))
        AC_luminance = solve_huffman_table('AC_luminance')
        f.writelines(str_hex_to_bytes('FFC4 00B5 10' + AC_luminance))
        DC_chrominance = solve_huffman_table('DC_chrominance')
        f.writelines(str_hex_to_bytes('FFC4 001F 01' + DC_chrominance))
        AC_chrominance = solve_huffman_table('AC_chrominance')
        f.writelines(str_hex_to_bytes('FFC4 00B5 11' + AC_chrominance))
        # picture data
        pic_data = 'FFDA 000C 03 0100 0211 0311 003F00' + data + 'FFD9'
        f.writelines(str_hex_to_bytes(pic_data))


def solve_huffman_table(table_name):
    """
    read the huffman table from the txt file
    :param table_name: AC_chrominance
    :return:
    """
    counter = [0 for i in range(17)]
    with open('./tables/%s.txt' % table_name) as f:
        tb = []
        for line in f.readlines():
            line = line.strip('\n')
            line = line.replace('/', '')
            tb.append(line.split('\t\t'))
    for each in tb:
        counter[len(each[1])] += 1
    counter = ''.join([hex(each)[2:].zfill(2) for each in counter[1:]])
    tb.sort(key=lambda x: x[1])
    out = ''.join([each[0].zfill(2) for each in tb])
    return counter + out


def str_hex_to_bytes(strings):
    """
    :param strings:"FF D8"
    :return: List:bytes of strings "1111 1111 1101 1000"
    """
    strings = strings.replace(' ', '')
    ret = []
    if len(strings) % 2 != 0:
        print("length of string is not mutiply of 2 %d %s" % (len(strings), strings))
        return []
    for i in range(0, len(strings) // 2):
        each = strings[2 * i:2 * i + 2]
        ret.append(pack('>B', int(each, 16)))
    return ret


def suit_len(byte_str):
    """
    fill the byte string into the mutiple of 8
    :param byte_str: "0111 0010 1101 10"
    :return: "0111 0010 1101 1000"
    """
    while len(byte_str) % 8 != 0:
        byte_str += '0'
    return byte_str


def main(bmp_path, jpeg_path):
    # read bmp picture
    bmp = ReadBMPFile(bmp_path)
    # get Y,Cb,Cr of the picture
    YCbCr = [bmp.Y, bmp.Cb, bmp.Cr]
    if bmp.biHeight % 8 != 0 or bmp.biWidth % 8 != 0:
        print("the width and height of the image should both be mutiples of 8")
        return
    # Split the image by 8x8 and do sampling, dct transfer, rounding the unit, 以8x8分割图像的Y,Cb,Cr,并且完成DCT变化,量化等操作
    split_YCbCr = []
    for index, each in enumerate(YCbCr):
        each_list = []
        pre_dc = 0
        for row in range(bmp.biHeight // 8):
            for column in range(bmp.biWidth // 8):
                # sampling
                unit = np.array(each[8 * row:8 * row + 8, 8 * column:8 * column + 8] - np.ones((8, 8)) * 128)
                # DCT transfer
                dct_unit = DCT(unit)
                # rounding
                round_unit = np.around(dct_unit / (Y_mat if index == 0 else CbCr_mat)).astype(int)
                # DC coefficient differ encode
                pre_dc, round_unit[0][0] = round_unit[0][0], round_unit[0][0] - pre_dc
                # transfer the matrix into list by zig_zag
                zig_zag_unit = tran_2d_1d(round_unit)
                # rel encoding to AC
                rel_unit = REL_encode(zig_zag_unit[1:])
                # byte encoding
                bit_unit = bit_encoding([(0, round_unit[0][0]), *rel_unit])
                each_list.append(bit_unit)
        split_YCbCr.append(each_list)

    # huffman encoding
    huffman_encoding = huffman_table()
    bit_list = []
    for index, each in enumerate(split_YCbCr):
        each_one = []
        for units in each:
            each_one.append(''.join(huffman_encoding.huffman_encoding_luminance(units))
                            if index == 0 else ''.join(huffman_encoding.huffman_encoding_chrominance(units)))
        bit_list.append(each_one)
    # package the Y,Cb,Cr into MCU
    MCU_list = [a + b + c for a, b, c in zip(bit_list[0], bit_list[1], bit_list[2])]
    data_byte_string = ''.join(MCU_list)
    data_hex_string = []
    data_byte_string = suit_len(data_byte_string)
    pre = None
    for i in range(0, len(data_byte_string) // 4):
        now = hex(int(data_byte_string[4 * i:4 * i + 4], 2))[2:]
        data_hex_string.append(now)
        if pre == 'f' and now == 'f' and i % 2 == 1:
            # once ‘ff’ appears, append ‘00’
            data_hex_string.append('00')
        pre = now
    write_file(''.join(data_hex_string), bmp.biHeight, bmp.biWidth, jpeg_path)


main(bmp_path='./lena512color.bmp', jpeg_path='ret.jpg')
