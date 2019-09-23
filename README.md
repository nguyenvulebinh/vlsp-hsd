## VLSP-SHARED Task: Hate Speech Detection on Social Networks

## Cấu trúc data
- File train.pkl

    Dict object của python, chứa toàn bộ dữ liệu train. Cấu trúc như sau:
    ```json
    {   
      "sample_id": {
        "raw": "raw sentence",
        "label": 1,
        "representation": {
          "comment": {
            "words": ["raw", "sent", "ence"],
            "vec": null 
          },
          "fasttext": {
            "words": ["raw", "sentence"],
            "vec": null 
          },
          "sonvx_wiki": {
            "words": ["raw", "sentence"],
            "vec": null 
          },
          "sonvx_baomoi_5": {
            "words": ["raw", "sentence"],
            "vec": null 
          },
          "sonvx_baomoi_2": {
            "words": ["raw", "sentence"],
            "vec": null 
          }
        }
      }
    }
    ```
    
- File test.pkl

    Dict object của python, chứa toàn bộ dữ liệu test. Cấu trúc giống file  train.pkl ngoại trừ việc không có label
    
- File dict_map.pkl
    
    Dict object của python, chứa mapping word với vector tương ứng với từng loại word embeding. Cấu trúc dict như sau:
    
    ```json
    {
      "comment": {
        "word_1": ndarray,
        "word_2": ndarray, 
        ...,
        "word_n": ndarray, 
      },
      "fasttext": {
        "word_1": ndarray,
        "word_2": ndarray, 
        ...,
        "word_n": ndarray, 
      },
      "sonvx_wiki": {
        "word_1": ndarray,
        "word_2": ndarray, 
        ...,
        "word_n": ndarray, 
      },
      "sonvx_baomoi_5": {
        "word_1": ndarray,
        "word_2": ndarray, 
        ...,
        "word_n": ndarray, 
      },
      "sonvx_baomoi_2": {
        "word_1": ndarray,
        "word_2": ndarray, 
        ...,
        "word_n": ndarray, 
      }
    }
    ```
    
    Hiện tại có 5 loại word embeding:
    
    - comment: skipgram train từ 370k comment (embedding size = 200)
    - fasttext: word embedding của fastext train cho dữ liệu tiếng Việt (embedding size = 300)
    - sonvx_wiki: word embedding train trên dữ liệu wiki của [sonvx](https://github.com/sonvx/word2vecVN) (embedding size = 400)
    - sonvx_wiki: word embedding train trên dữ liệu baomoi với window_size = 5 của [sonvx](https://github.com/sonvx/word2vecVN) (embedding size = 400)
    - sonvx_wiki: word embedding train trên dữ liệu baomoi với window_size = 5 của [sonvx](https://github.com/sonvx/word2vecVN) (embedding size = 300)
    
    
## Cách sử dụng dữ liệu đã qua xử lý

Sử dụng hàm load_dataset.py. Hàm này khi được import sẽ load sẵn các file train.pkl, test.pkl và dict_map.pkl. Đồng thời chia 
luôn train và valid dựa trên dữ liệu từ file train.pkl. Tập valid có danh sách sample id lưu trong biến valid_data_ids. 
Để load các tập train valid và test này sử dụng các hàm sau:
- generate_batches_train(batch_size, embed_type) Load dữ liệu train, trả về iterator để chạy theo batch. Mỗi step sẽ trả về cặp input và output
- generate_batches_train(batch_size, embed_type, ids=valid_data_ids) Load dữ liệu valid, trả về iterator để chạy theo batch. Mỗi step sẽ trả về cặp input và output
- generate_batches_test(batch_size, embed_type)  Load dữ liệu test, trả về iterator để chạy theo batch. Mỗi step sẽ trả về cặp input và sample_id

embed_type là một trong những loại sau ["comment", "fasttext", "sonvx_wiki", "sonvx_baomoi_5", "sonvx_baomoi_2"]