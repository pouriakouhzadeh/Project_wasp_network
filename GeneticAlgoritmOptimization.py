import random
import logging
from deap import base, creator, tools
from celery import group
import pandas as pd
import pickle
from tasks import train_model_task
from TR_MODEL import TrainModels
from TR_MODEL_MODEL import TrainModelsReturn
import os

# تنظیمات logging برای ذخیره لاگ‌ها در فایل و نمایش در کنسول
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[
                        logging.FileHandler("genetic_algorithm.log"),
                        logging.StreamHandler()
                    ])

def check_directories():
    paths = [
        "/home/pouria/project/trained_models/",
        "/home/pouria/project/csv_files_initial/",
        "/home/pouria/project/temp_csv_dir/"
    ]
    for path in paths:
        if not os.path.exists(path):
            logging.error(f"Path does not exist: {path}")
        if not os.access(path, os.W_OK):
            logging.error(f"No write access to path: {path}")

class GeneticAlgorithm:
    def __init__(self):
        self.TR = TrainModels()
        self.TRMM = TrainModelsReturn()
        self._initialize_deap()
        self.toolbox = base.Toolbox()
        self.param_space = [(2, 12), (2, 30), (30, 800), (12000, 20000), (1000, 3000), (52, 75), (12, 18)]
        self.toolbox.register("individual", self.generate_individual, self.param_space)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.create_task_for_individual)
        self.toolbox.register("mate", self.custom_mate)
        self.toolbox.register("mutate", self.custom_mutate, low=0, up=10, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def custom_mate(self, ind1, ind2):
        size = len(ind1)
        cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        self.check_bounds(ind1)
        self.check_bounds(ind2)
        return ind1, ind2

    def custom_mutate(self, individual, low, up, indpb):
        size = len(individual)
        for i in range(size):
            if random.random() < indpb:
                individual[i] = random.randint(low, up)
        self.check_bounds(individual)
        return individual

    def check_bounds(self, individual):
        for i, param in enumerate(self.param_space):
            if isinstance(individual[i], list):
                individual[i] = [x if param[0] <= x <= param[1] else random.randint(param[0], param[1]) for x in individual[i]]
            else:
                individual[i] = max(param[0], min(param[1], individual[i]))

    def _initialize_deap(self):
        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def df_to_json(self, df):
        # Setting double precision to 15 to maintain floating point accuracy
        return df.to_json(orient='split', double_precision=15)

    def json_to_df(self, json_data):
        return pd.read_json(json_data, orient='split')

    def save_to_file(self, data, file_name="ga_results.txt"):
        with open(file_name, "a") as file:
            file.write(data + "\n")

    def read_data(self, file_name, tail_size=21000):
        file_path = f"/home/pouria/project/temp_csv_dir/{file_name}"
        try:
            data = pd.read_csv(file_path)
            data = data.tail(tail_size)
            data.reset_index(inplace=True, drop=True)
            return data
        except FileNotFoundError:
            logging.error(f"File {file_name} not found.")
            return None

    def create_task_for_individual(self, individual, currency_pair):
        currency_data = self.read_data(f"{currency_pair}.csv", 21000)
        if currency_data is None:
            return None
        # logging.info(f"Creating task for individual: {individual}")
        return train_model_task.s(self.df_to_json(currency_data), *individual[:-1], individual[-1])

    def generate_individual(self, param_space):
        individual = []
        for param in param_space[:-1]:
            if isinstance(param[0], (bool, str)):
                individual.append(random.choice(param))
            else:
                try:
                    if param[0] > param[1]:
                        raise ValueError(f"Invalid range: ({param[0]}, {param[1]})")
                    individual.append(random.randint(param[0], param[1]))
                except ValueError as e:
                    logging.error(f"Error with parameter range: {e}")
                    return None

        last_param = param_space[-1]
        if last_param[0] > last_param[1]:
            logging.error(f"Invalid range for hours: ({last_param[0]}, {last_param[1]})")
            return None

        allowed_hours_count = random.randint(last_param[0], last_param[1])
        allowed_hours_set = set()

        while len(allowed_hours_set) < allowed_hours_count:
            hour = random.randint(3, 23)
            if hour not in allowed_hours_set:
                allowed_hours_set.add(hour)

        individual.append(list(allowed_hours_set))
        # logging.info(f"Generated individual: {individual}")
        return creator.Individual(individual)

    def main(self, currency_pair, pop, NG):
        check_directories()  # بررسی مسیرها و دسترسی‌ها قبل از شروع
        population = self.toolbox.population(n=pop)
        CXPB, MUTPB, NGEN = 0.5, 0.1, NG
        ELITISM_SIZE = 5

        tasks = group(self.create_task_for_individual(ind, currency_pair) for ind in population)
        results = tasks.apply_async()
        fitnesses = results.get()

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        logging.info(f"{currency_pair} >>> Population: {pop} & Generation: 0/{NG} & Best fitness = {max(fitnesses)}")

        for g in range(NGEN):
            offspring = self.toolbox.select(population, len(population) - ELITISM_SIZE)
            elites = tools.selBest(population, ELITISM_SIZE)
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            tasks = group(self.create_task_for_individual(ind, currency_pair) for ind in invalid_ind)
            results = tasks.apply_async()
            fitnesses = results.get()

            try:
                Max_ = max(fitnesses)
            except Exception as e:
                Max_ = 0
                logging.error(f"Error calculating max fitness: {e}")
            logging.info(f"{currency_pair} >>> Population: {pop} & Generation: {g}/{NG} & Best fitness = {Max_}")


            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Check and save individuals with fitness >= 0.8



            best_individuals = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)[:5]
            if Max_ > 0.8 :
                logging.info(f"{currency_pair} >>>> start try 5 best fitness for this model")
                for i, ind in enumerate(best_individuals, 1):
                    logging.info(f"Top {i} Individual = {ind}, Fitness = {ind.fitness.values[0]}")
                    data = self.read_data(f"{currency_pair}.csv", 21000)
                    df1 = self.df_to_json(data)
                    df2 = self.json_to_df(df1)
                    model1, feature_indicts1, scaler1, acc1= 0, 0, 0, 0
                    model1, feature_indicts1, scaler1, acc1 = self.TRMM.Train(df2.copy(), *ind[:-1], ind[-1])
                    if model1 != 0 and feature_indicts1 != 0 and scaler1 != 0 and acc1 != 0 :
                        logging.info("Model, feature_indicts, scaler are valid, attempting to save...")
                        try:
                            with open(f"/home/pouria/project/trained_models/{currency_pair}.pkl", 'wb') as file:
                                pickle.dump(model1, file)
                            with open(f"/home/pouria/project/trained_models/{currency_pair}_parameters.pkl", 'wb') as file:
                                pickle.dump(ind, file)
                            with open(f"/home/pouria/project/trained_models/{currency_pair}_features_indicts.pkl", 'wb') as file:
                                pickle.dump(feature_indicts1, file)
                                logging.info(f"feature indicts1 : {feature_indicts1}")
                            with open(f"/home/pouria/project/trained_models/{currency_pair}_scaler.pkl", 'wb') as file:
                                pickle.dump(scaler1, file)
                            currency_data_for_save = self.read_data(f"{currency_pair}.csv", tail_size=ind[3])
                            file_path = f"/home/pouria/project/csv_files_initial/{currency_pair}.csv"
                            currency_data_for_save.to_csv(file_path, index=False)
                            logging.info(f"Model {currency_pair} saved successfully with fitness {ind.fitness.values[0]}")
                            raise Exception(f"Model {currency_pair} saved successfully...")
                        except Exception as e:
                            logging.error(f"Error saving model for {currency_pair}: {e}")
                            # raise e  # پرتاب استثنا برای تلاش دوباره توسط Celery
                    else:
                        logging.error(f"Model {currency_pair}  with acc : {acc1} >>>> {i}/5 is not grather than 0.78")
                        # raise Exception(f"Model {currency_pair} not saved due to training failure")  # پرتاب استثنا برای تلاش دوباره توسط Celery

            offspring.extend(elites)
            population[:] = offspring


        logging.info(f"Model for {currency_pair} saving failed")
        ind[0] = 0
        ind[1] = 0
        ind[2] = 0
        with open(f"/home/pouria/project/trained_models/{currency_pair}_parameters.pkl", 'wb') as file:
            pickle.dump(ind, file)
        raise Exception(f"Model for {currency_pair} saving failed")  # پرتاب استثنا برای تلاش دوباره توسط Celery

if __name__ == "__main__":
    check_directories()  # بررسی مسیرها و دسترسی‌ها قبل از شروع برنامه اصلی
    ga = GeneticAlgorithm()

