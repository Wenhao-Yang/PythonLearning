#ifndef DEMO_H
#define DEMO_H
using namespace std;
namespace demo {
    class MyDemo {
        public:
            int a;
            MyDemo();
            MyDemo(int a );
            ~MyDemo();
            int mul(int m );
            int add(int b);
            void sayHello(char* name);
    };
}
#endif