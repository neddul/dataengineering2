import pulsar

def consume_and_process(subscription_name):
    client = pulsar.Client('pulsar://localhost:6650')
    consumer = client.subscribe('word-topic', subscription_name)
    producer = client.create_producer('result-topic')

    try:
        while True:
            msg = consumer.receive()
            processed_word = msg.data().decode('utf-8').upper()
            print(f"Processing: {processed_word}")
            producer.send(processed_word.encode('utf-8'))
            consumer.acknowledge(msg)
    except Exception as e:
        print("Error:", e)
    finally:
        consumer.close()
        producer.close()
        client.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python consumer.py <subscription_name>")
        sys.exit(1)
    subscription_name = sys.argv[1]
    consume_and_process(subscription_name)
