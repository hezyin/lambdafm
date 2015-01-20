#!/usr/bin/env ruby

ps = []
1.upto(3) do |i|
  1.upto(3) do |j|
    1.upto(3) do |k|
      ps << [i * 0.001, j * 0.001, k * 0.001]
    end
  end
end

start = Time.now
f = open("ptuner_ruby_log", "w+")

ps.each_with_index do |p, i|
  f.puts "#{i+1}/#{total} k=16 e0=0.004 e1=#{p[0]} e2=#{p[1]} e3=#{p[2]} l=0.00002 #{(Time.now - start)/60}mins" 
  `./lambdafm -k 16 -t 10 -s 8 -e1 #{p[0]} -e2 #{p[1]} -e3 #{p[2]} -l 0.00002 -g k_16_e1_#{p[0]}_e2_#{p[1]}_e3_#{p[2]}_l_0.00002 train.libfm test.libfm`
end

f.close
