import h5py
import numpy as np

def generate_hdf5():
    # Features: age, income, is_married (protected), target (approved)
    # Plus some correlated/useless features
    np.random.seed(42)
    n = 1000
    
    age = np.random.normal(40, 10, n)
    income = age * 1.5 + np.random.normal(0, 5, n)  # correlated with age
    
    # 0 = Unmarried, 1 = Married
    is_married = np.random.choice([0, 1], n)
    
    # Target heavily influenced by income and is_married representing bias
    score = income * 0.1 + is_married * 5 + np.random.normal(0, 2, n)
    target = (score > np.median(score)).astype(int)
    
    # Text col for PII test
    names = ["John Doe", "Jane Smith", "Will User", "Alice", "Bob"]
    name_col = np.random.choice(names, n).astype('S')
    
    # Combine into structured array
    dt = np.dtype([('age', 'f4'), ('income', 'f4'), ('is_married', 'i4'), ('approved', 'i4'), ('name', 'S20')])
    data = np.empty(n, dtype=dt)
    
    data['age'] = age
    data['income'] = income
    data['is_married'] = is_married
    data['approved'] = target
    data['name'] = name_col
    
    with h5py.File("test_data.h5", "w") as f:
        f.create_dataset("dataset1", data=data)
        
if __name__ == "__main__":
    generate_hdf5()
    print("Generated test_data.h5")
