// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import junit.framework.TestCase;

public class MonteCarloAlgorithmsTest extends TestCase {
  public void testExamplesWithSarsa() {
    checkExampleWithSarsa(MonteCarloExamples.AIRPORT);
    checkExampleWithSarsa(MonteCarloExamples.CLIFFWALK);
    checkExampleWithSarsa(MonteCarloExamples.GAMBLER_20);
    checkExampleWithSarsa(MonteCarloExamples.GRIDWORLD);
    checkExampleWithSarsa(MonteCarloExamples.INFINITEVARIANCE);
    checkExampleWithSarsa(MonteCarloExamples.MAXBIAS);
    checkExampleWithSarsa(MonteCarloExamples.MAZE2);
    checkExampleWithSarsa(MonteCarloExamples.RACETRACK);
    checkExampleWithSarsa(MonteCarloExamples.WINDYGRID);
    checkExampleWithSarsa(MonteCarloExamples.WIRELOOP_4);
    checkExampleWithSarsa(MonteCarloExamples.WIRELOOP_C);
  }

  private static void checkExampleWithSarsa(MonteCarloExamples example) {
    System.out.println("Testing: " + example.toString());
    int batches = 5;
    DiscreteQsa optimalQsa = MonteCarloAnalysis.getOptimalQsa(example.get(), batches);
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.ORIGINAL_SARSA);
    list.add(MonteCarloAlgorithms.EXPECTED_SARSA);
    list.add(MonteCarloAlgorithms.QLEARNING_SARSA);
    list.add(MonteCarloAlgorithms.DOUBLE_QLEARNING_SARSA);
    list.add(MonteCarloAlgorithms.TRUE_ONLINE_SARSA);
    // ---
    List<DiscreteModelErrorAnalysis> errorAnalysis = new ArrayList<>();
    errorAnalysis.add(DiscreteModelErrorAnalysis.LINEAR_POLICY);
    // ---
    MonteCarloInterface monteCarloInterface = example.get();
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      monteCarloAlgorithms.analyseNTimes(monteCarloInterface, batches, optimalQsa, errorAnalysis, 1);
  }

  public void testExamplesWithSeveralTrials() {
    MonteCarloExamples example = MonteCarloExamples.AIRPORT;
    System.out.println("Testing: " + example.toString());
    int batches = 5;
    DiscreteQsa optimalQsa = MonteCarloAnalysis.getOptimalQsa(example.get(), batches);
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.ORIGINAL_SARSA);
    list.add(MonteCarloAlgorithms.DOUBLE_QLEARNING_SARSA);
    list.add(MonteCarloAlgorithms.TRUE_ONLINE_SARSA);
    list.add(MonteCarloAlgorithms.MONTE_CARLO);
    // ---
    List<DiscreteModelErrorAnalysis> errorAnalysis = new ArrayList<>();
    errorAnalysis.add(DiscreteModelErrorAnalysis.LINEAR_POLICY);
    // ---
    MonteCarloInterface monteCarloInterface = example.get();
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      monteCarloAlgorithms.analyseNTimes(monteCarloInterface, batches, optimalQsa, errorAnalysis, 10);
  }

  public void testExamplesWithSeveralErrorAnalysis() {
    MonteCarloExamples example = MonteCarloExamples.AIRPORT;
    System.out.println("Testing: " + example.toString());
    int batches = 5;
    DiscreteQsa optimalQsa = MonteCarloAnalysis.getOptimalQsa(example.get(), batches);
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.ORIGINAL_SARSA);
    list.add(MonteCarloAlgorithms.DOUBLE_QLEARNING_SARSA);
    list.add(MonteCarloAlgorithms.TRUE_ONLINE_SARSA);
    list.add(MonteCarloAlgorithms.MONTE_CARLO);
    // ---
    List<DiscreteModelErrorAnalysis> errorAnalysis = new ArrayList<>();
    errorAnalysis.add(DiscreteModelErrorAnalysis.LINEAR_POLICY);
    errorAnalysis.add(DiscreteModelErrorAnalysis.LINEAR_QSA);
    // ---
    MonteCarloInterface monteCarloInterface = example.get();
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      monteCarloAlgorithms.analyseNTimes(monteCarloInterface, batches, optimalQsa, errorAnalysis, 2);
  }

  public void testExamplesWithMC() {
    checkExampleWithMC(MonteCarloExamples.AIRPORT);
    checkExampleWithMC(MonteCarloExamples.GAMBLER_20);
    checkExampleWithMC(MonteCarloExamples.INFINITEVARIANCE);
    checkExampleWithMC(MonteCarloExamples.MAXBIAS);
    checkExampleWithMC(MonteCarloExamples.MAZE2);
    checkExampleWithMC(MonteCarloExamples.RACETRACK);
  }

  private static void checkExampleWithMC(MonteCarloExamples example) {
    System.out.println("Testing: " + example.toString());
    int batches = 10;
    DiscreteQsa optimalQsa = MonteCarloAnalysis.getOptimalQsa(example.get(), batches);
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.MONTE_CARLO);
    // ---
    List<DiscreteModelErrorAnalysis> errorAnalysis = new ArrayList<>();
    errorAnalysis.add(DiscreteModelErrorAnalysis.LINEAR_POLICY);
    // ---
    MonteCarloInterface monteCarloInterface = example.get();
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      monteCarloAlgorithms.analyseNTimes(monteCarloInterface, batches, optimalQsa, errorAnalysis, 1);
  }

  public void testVirtualStationExample() {
    MonteCarloExamples example = MonteCarloExamples.VIRTUALSTATIONS;
    System.out.println("Testing: " + example.toString());
    int batches = 1;
    DiscreteQsa optimalQsa = MonteCarloAnalysis.getOptimalQsa(example.get(), batches);
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.MONTE_CARLO);
    list.add(MonteCarloAlgorithms.EXPECTED_SARSA);
    list.add(MonteCarloAlgorithms.QLEARNING_SARSA);
    list.add(MonteCarloAlgorithms.DOUBLE_QLEARNING_SARSA);
    list.add(MonteCarloAlgorithms.TRUE_ONLINE_SARSA);
    // ---
    List<DiscreteModelErrorAnalysis> errorAnalysis = new ArrayList<>();
    errorAnalysis.add(DiscreteModelErrorAnalysis.LINEAR_POLICY);
    // ---
    MonteCarloInterface monteCarloInterface = example.get();
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      monteCarloAlgorithms.analyseNTimes(monteCarloInterface, batches, optimalQsa, errorAnalysis, 1);
  }
}