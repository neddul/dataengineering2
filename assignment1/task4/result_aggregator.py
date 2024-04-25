import pulsar

def aggregate_results():
    client = pulsar.Client('pulsar://localhost:6650')
    consumer = client.subscribe('result-topic', 'aggregator-subscription')
    final_result = []

    try:
        for i in range(5):  # assuming we know there are 5 words
            msg = consumer.receive()
            final_result.append(msg.data().decode('utf-8'))
            consumer.acknowledge(msg)
    except Exception as e:
        print("Error:", e)
    finally:
        consumer.close()
        client.close()

    print("Final result:", " ".join(final_result))

if __name__ == "__main__":
    aggregate_results()
