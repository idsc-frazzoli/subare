// code by jph
package ch.ethz.idsc.subare.demo.prison;

import ch.ethz.idsc.subare.ch02.Agent;

class SummaryString {
  public static String of(Agent agent) {
    int rnd = agent.getRandomizedDecisionCount();
    double avg = agent.getRewardAverage().number().doubleValue();
    return String.format("%25s  %6.3f  %5d RND", //
        agent.toString(), avg, rnd);
  }
}
