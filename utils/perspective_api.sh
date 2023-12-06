python perspective_api_span.py \
--file ../dataset/span_cnn_train.json \
--output ../dataset/span_cnn_train_score.json \
--api_key your-api-key \
--api_rate 90000 \
--process 1

python perspective_api_span.py \
--file ../dataset/span_cnn_test.json \
--output ../dataset/span_cnn_test_score.json \
--api_key your-api-key \
--api_rate 9000 \
--process 1