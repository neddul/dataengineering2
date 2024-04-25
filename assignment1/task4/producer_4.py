import pulsar

def produce_messages():
    client = pulsar.Client('pulsar://localhost:6650')
    producer = client.create_producer('word-topic')

    input_string = "I want to be capatilized"
    words = input_string.split()

    for word in words:
        producer.send((word.encode('utf-8')))

    producer.close()
    client.close()

if __name__ == "__main__":
    produce_messages()
