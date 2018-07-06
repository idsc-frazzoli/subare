package ch.ethz.idsc.subare.analysis;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import junit.framework.TestCase;

public class AnalysisTest extends TestCase {
  public void testExamplesWithSarsa() {
    testExampleWithSarsa(MonteCarloExamples.AIRPORT);
    testExampleWithSarsa(MonteCarloExamples.CLIFFWALK);
    // testExample(MonteCarloExamples.GAMBLER_100);
    testExampleWithSarsa(MonteCarloExamples.GAMBLER_20);
    testExampleWithSarsa(MonteCarloExamples.GRIDWORLD);
    testExampleWithSarsa(MonteCarloExamples.INFINITEVARIANCE);
    testExampleWithSarsa(MonteCarloExamples.MAXBIAS);
    testExampleWithSarsa(MonteCarloExamples.MAZE2);
    // testExample(MonteCarloExamples.MAZE5);
    testExampleWithSarsa(MonteCarloExamples.RACETRACK);
    testExampleWithSarsa(MonteCarloExamples.WINDYGRID);
    testExampleWithSarsa(MonteCarloExamples.WIRELOOP_4);
    // testExampleWithSarsa(MonteCarloExamples.WIRELOOP_5);
    testExampleWithSarsa(MonteCarloExamples.WIRELOOP_C);
  }

  private void testExampleWithSarsa(MonteCarloExamples example) {
    System.out.println("Testing: " + example.toString());
    int batches = 5;
    DiscreteQsa optimalQsa = MonteCarloAnalysis.getOptimalQsa(example.get(), batches);
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.OriginalSarsa);
    list.add(MonteCarloAlgorithms.ExpectedSarsa);
    list.add(MonteCarloAlgorithms.QLearningSarsa);
    list.add(MonteCarloAlgorithms.DoubleQLearningSarsa);
    list.add(MonteCarloAlgorithms.TrueOnlineSarsa);
    list.add(MonteCarloAlgorithms.TrueOnlineSarsaWarmStart);
    // ---
    MonteCarloErrorAnalysis errorAnalysis = MonteCarloErrorAnalysis.LINEAR_POLICY;
    // ---
    MonteCarloInterface monteCarloInterface = example.get();
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      monteCarloAlgorithms.analyse(monteCarloInterface, batches, optimalQsa, errorAnalysis);
  }

  public void testExamplesWithMC() {
    testExampleWithMC(MonteCarloExamples.AIRPORT);
    testExampleWithMC(MonteCarloExamples.GAMBLER_20);
    testExampleWithMC(MonteCarloExamples.INFINITEVARIANCE);
    testExampleWithMC(MonteCarloExamples.MAXBIAS);
    testExampleWithMC(MonteCarloExamples.MAZE2);
    testExampleWithMC(MonteCarloExamples.RACETRACK);
  }

  private void testExampleWithMC(MonteCarloExamples example) {
    System.out.println("Testing: " + example.toString());
    int batches = 10;
    DiscreteQsa optimalQsa = MonteCarloAnalysis.getOptimalQsa(example.get(), batches);
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.MonteCarlo);
    // ---
    MonteCarloErrorAnalysis errorAnalysis = MonteCarloErrorAnalysis.LINEAR_POLICY;
    // ---
    MonteCarloInterface monteCarloInterface = example.get();
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      monteCarloAlgorithms.analyse(monteCarloInterface, batches, optimalQsa, errorAnalysis);
  }

  public void testVirtualStationExample() {
    MonteCarloExamples example = MonteCarloExamples.VIRTUALSTATIONS;
    System.out.println("Testing: " + example.toString());
    int batches = 1;
    DiscreteQsa optimalQsa = MonteCarloAnalysis.getOptimalQsa(example.get(), batches);
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.MonteCarlo);
    list.add(MonteCarloAlgorithms.ExpectedSarsa);
    list.add(MonteCarloAlgorithms.QLearningSarsa);
    list.add(MonteCarloAlgorithms.DoubleQLearningSarsa);
    list.add(MonteCarloAlgorithms.TrueOnlineSarsa);
    list.add(MonteCarloAlgorithms.TrueOnlineSarsaWarmStart);
    // ---
    MonteCarloErrorAnalysis errorAnalysis = MonteCarloErrorAnalysis.LINEAR_POLICY;
    // ---
    MonteCarloInterface monteCarloInterface = example.get();
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      monteCarloAlgorithms.analyse(monteCarloInterface, batches, optimalQsa, errorAnalysis);
  }
}