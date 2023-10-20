import g4f
import g4f.api

if __name__ == "__main__":
    print(f'Starting server... [g4f v-{g4f.version}]')
    g4f.api.Api(g4f).run('localhost:1337', 8)