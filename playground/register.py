agent_registry = {}

def register_agent(name):
    def wrapper(cls):
        agent_registry[name] = cls
        return cls
    return wrapper
