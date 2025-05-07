#include <iostream>
#include <omp.h>
using namespace std;

// Function to swap two integers
void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// Parallel Bubble Sort (Odd-Even Transposition Sort)
void bubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        int first = i % 2;

        #pragma omp parallel for shared(a, first)
        for (int j = first; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

int main() {
    int *a, n;

    cout << "\nEnter total number of elements: ";
    cin >> n;

    a = new int[n];
    cout << "\nEnter elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }

    bubble(a, n);

    cout << "\nSorted array is:\n";
    for (int i = 0; i < n; i++) {
        cout << a[i] << endl;
    }

    delete[] a; // Free the allocated memory
    return 0;
}


